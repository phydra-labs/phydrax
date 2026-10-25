"""Small AIMNet2 source inventories and an independent NumPy source reference.

The inventory uses the AIMNetcentral 0.2.0 state-dictionary names and layouts
(``nn.Linear`` weights ``(out, in)``, ``Sequential`` indices, ``ModuleDict``
output names). ``reference_forward`` re-implements the source dense forward
(``nb_mode == 0``, or its lattice-image pair sums for one periodic cell) directly
from the source formulas with NumPy, reading only the inventory; it shares no
code with the native implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np


SPECIES = (1, 6, 7, 8)
FEATURES = 3
SHIFTS = 4
COMBINATIONS = 2
HIDDEN = [[6, 5], [5], [4, 3]]
AIM = 4
RC_S = 3.2
RC_SR = 2.4
TS = {"a1": 0.4, "a2": 3.0, "s8": 0.8, "s6": 1.0}
HARTREE = 27.211386024367243
BOHR = 0.5291772105638411


def _declared[T](value: T | None, stage: str) -> T:
    """A source value an output stage consumes; declaration order must produce it."""
    if value is None:
        raise ValueError(
            f"Source output {stage} consumes a value no earlier stage produced."
        )
    return value


def _gelu(value: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * value * (1.0 + erf(value / math.sqrt(2.0)))


def _activation(path: str, options: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """``torch.nn`` activation semantics written independently with NumPy."""
    name = path.rsplit(".", 1)[-1]
    if name == "GELU":
        if options.get("approximate", "none") == "tanh":
            return lambda x: (
                0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
            )
        return _gelu
    if name == "SiLU":
        return lambda x: x / (1 + np.exp(-x))
    if name == "Softplus":
        beta = options.get("beta", 1.0)
        threshold = options.get("threshold", 20.0)
        return lambda x: np.where(
            beta * x > threshold,
            x,
            np.log1p(np.exp(np.minimum(beta * x, threshold))) / beta,
        )
    raise AssertionError(path)


def source_inventory(
    seed: int,
    *,
    d2: bool = True,
    channels: int = 1,
    dispersion: bool = False,
    moments: bool = False,
    center: bool = False,
    envelope: str = "exp",
    head_activation: tuple[str, dict[str, Any]] | None = None,
    frozen_shift: bool = False,
    repulsion: bool = False,
    long_range: bool = False,
    scale: float = 0.3,
    aev_cutoff: float = RC_S,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """A random small AIMNet2 declaration and its exact state inventory.

    ``long_range`` embeds ``LRCoulomb`` (simple, ``subtract_sr``) in place of
    ``SRCoulomb``, as source training configurations do; ``repulsion`` adds an
    energy-bearing ``core.SRRep`` with ``exp_cutoff`` at 2.8 A. ``aev_cutoff``
    is the declared ``aev.rc_s``; the source defaults derive ``rc_v``, ``eta``
    and the shift grid from it, while the ``SRCoulomb``/``LRCoulomb`` radius
    ``RC_SR`` and the ``SRRep`` radius stay their own declared values.
    """
    if not math.isfinite(aev_cutoff) or aev_cutoff <= 0.8:
        raise ValueError(
            "aev_cutoff must exceed the source shift-grid minimum rmin = 0.8 A."
        )
    rng = np.random.default_rng(seed)
    total = FEATURES * SHIFTS if d2 else FEATURES
    tensors: dict[str, np.ndarray] = {}
    eta = (1.0 / ((aev_cutoff - 0.8) / SHIFTS)) ** 2
    shifts = np.linspace(0.8, aev_cutoff, SHIFTS + 1)[:SHIFTS]
    for suffix in ("s", "v"):
        tensors[f"aev.rc_{suffix}"] = np.asarray(aev_cutoff)
        tensors[f"aev.eta_{suffix}"] = np.asarray(eta)
        tensors[f"aev.shifts_{suffix}"] = shifts.copy()
    afv = np.full((64, total), np.nan)
    afv[0] = 0.0
    afv[list(SPECIES)] = rng.normal(size=(len(SPECIES), total))
    tensors["afv.weight"] = afv
    tensors["conv_a.agh"] = rng.normal(size=(FEATURES, SHIFTS, COMBINATIONS))
    tensors["conv_q.agh"] = rng.normal(size=(channels, SHIFTS, COMBINATIONS))
    conv_a = FEATURES * (SHIFTS + COMBINATIONS)
    conv_q = channels * (SHIFTS + COMBINATIONS)
    first = total + conv_a
    later = first + channels + conv_q
    update = total + 2 * channels
    passes = [
        (first, HIDDEN[0], update),
        (later, HIDDEN[1], update),
        (later, HIDDEN[2], AIM),
    ]
    for index, (inputs, hidden, outputs) in enumerate(passes):
        sizes = [inputs, *hidden, outputs]
        for layer in range(len(sizes) - 1):
            tensors[f"mlps.{index}.{2 * layer}.weight"] = scale * rng.normal(
                size=(sizes[layer + 1], sizes[layer])
            )
            tensors[f"mlps.{index}.{2 * layer}.bias"] = scale * rng.normal(
                size=(sizes[layer + 1],)
            )

    def head(name: str, hidden: list[int], outputs: int) -> None:
        sizes = [AIM, *hidden, outputs]
        for layer in range(len(sizes) - 1):
            tensors[f"outputs.{name}.mlp.{2 * layer}.weight"] = scale * rng.normal(
                size=(sizes[layer + 1], sizes[layer])
            )
            tensors[f"outputs.{name}.mlp.{2 * layer}.bias"] = scale * rng.normal(
                size=(sizes[layer + 1],)
            )

    outputs: dict[str, Any] = {
        "energy_mlp": {
            "class": "aimnet.modules.Output",
            "kwargs": {
                "n_in": AIM,
                "n_out": 1,
                "key_in": "aim",
                "key_out": "energy",
                "mlp": {
                    "activation_fn": "torch.nn.GELU"
                    if head_activation is None
                    else head_activation[0],
                    "activation_kwargs": {}
                    if head_activation is None
                    else head_activation[1],
                    "last_linear": True,
                    "hidden": [5],
                },
            },
        }
    }
    head("energy_mlp", [5], 1)
    if dispersion:
        outputs["disp_head"] = {
            "class": "aimnet.modules.Output",
            "kwargs": {
                "n_in": AIM,
                "n_out": 2,
                "key_in": "aim",
                "key_out": "disp_param",
                "mlp": {"activation_fn": "torch.nn.GELU", "hidden": [3]},
            },
        }
        head("disp_head", [3], 2)
    outputs["atomic_shift"] = {
        "class": "aimnet.modules.AtomicShift",
        "kwargs": {
            "key_in": "energy",
            "key_out": "energy",
            **({"requires_grad": False} if frozen_shift else {}),
        },
    }
    shifts_table = rng.normal(size=(64, 1)) * 3.0
    shifts_table[0] = 0.0
    tensors["outputs.atomic_shift.shifts.weight"] = shifts_table
    outputs["atomic_sum"] = {
        "class": "aimnet.modules.AtomicSum",
        "kwargs": {"key_in": "energy", "key_out": "energy"},
    }
    masses = np.linspace(0.0, 120.0, 119)
    if moments:
        for kind in ("dipole", "quadrupole"):
            outputs[kind] = {
                "class": f"aimnet.modules.{kind.capitalize()}",
                "kwargs": {"key_in": "charges", "key_out": kind, "center_coord": center},
            }
            tensors[f"outputs.{kind}.mass"] = masses.copy()
    else:
        # Older artifacts carry these buffers without the modules.
        tensors["outputs.dipole.mass"] = masses.copy()
    if long_range:
        outputs["lrcoulomb"] = {
            "class": "aimnet.modules.LRCoulomb",
            "kwargs": {
                "rc": RC_SR,
                "key_in": "charges",
                "key_out": "energy",
                "method": "simple",
                "envelope": envelope,
            },
        }
        tensors["outputs.lrcoulomb.rc"] = np.asarray(RC_SR)
    else:
        outputs["srcoulomb"] = {
            "class": "aimnet.modules.SRCoulomb",
            "kwargs": {
                "rc": RC_SR,
                "key_in": "charges",
                "key_out": "energy",
                "envelope": envelope,
            },
        }
        tensors["outputs.srcoulomb.rc"] = np.asarray(RC_SR)
    if repulsion:
        outputs["srrep"] = {
            "class": "aimnet.modules.core.SRRep",
            "kwargs": {"key_out": "energy", "cutoff_fn": "exp_cutoff", "rc": 2.8},
        }
        tensors["outputs.srrep.rc"] = np.asarray(2.8)
        parameters = np.zeros((87, 2))
        parameters[1:] = rng.uniform(0.3, 1.5, size=(86, 2))
        tensors["outputs.srrep.params.weight"] = parameters
    if dispersion:
        outputs["dispparam"] = {
            "class": "aimnet.modules.DispParam",
            "kwargs": {"key_in": "disp_param", "key_out": "disp_param"},
        }
        reference = np.zeros((87, 2))
        reference[0] = (0.0, 1.0)
        reference[list(SPECIES), 0] = rng.uniform(2.0, 40.0, size=len(SPECIES))
        reference[list(SPECIES), 1] = rng.uniform(4.0, 12.0, size=len(SPECIES))
        tensors["outputs.dispparam.disp_param0"] = reference
        outputs["d3ts"] = {"class": "aimnet.modules.D3TS", "kwargs": dict(TS)}
        # Synthetic diagnostic buffer, not a bundled elemental reference table.
        radii = np.linspace(1.0, 4.0, 119, dtype=np.float64)
        radii[0] = 0.0
        tensors["outputs.d3ts.r4r2"] = radii
    declaration = {
        "class": "aimnet.models.AIMNet2",
        "kwargs": {
            "nfeature": FEATURES,
            "d2features": d2,
            "ncomb_v": COMBINATIONS,
            "hidden": HIDDEN,
            "aim_size": AIM,
            "num_charge_channels": channels,
            "aev": {"rc_s": aev_cutoff, "nshifts_s": SHIFTS},
            "outputs": outputs,
        },
    }
    return declaration, tensors


def _mlp(
    tensors: dict[str, np.ndarray],
    prefix: str,
    value: np.ndarray,
    last_linear: bool,
    activation: Callable[[np.ndarray], np.ndarray] = _gelu,
) -> np.ndarray:
    layer = 0
    while f"{prefix}.{2 * layer}.weight" in tensors:
        layer += 1
    for index in range(layer):
        value = (
            value @ tensors[f"{prefix}.{2 * index}.weight"].T
            + tensors[f"{prefix}.{2 * index}.bias"]
        )
        if not (last_linear and index == layer - 1):
            value = activation(value)
    return value


def _periodic_reach(
    outputs: dict[str, Any], tensors: dict[str, np.ndarray], rc: float, ts_cutoff: float
) -> float:
    """The largest pair radius of a periodic forward; unbounded sums are refused."""
    radii = [rc]
    for name, entry in outputs.items():
        kind = entry["class"].rsplit(".", 1)[-1]
        if kind in ("SRCoulomb", "SRRep"):
            radii.append(float(tensors[f"outputs.{name}.rc"]))
        elif kind == "D3TS":
            if not math.isfinite(ts_cutoff):
                raise ValueError("A periodic D3TS sum needs a finite ts_cutoff.")
            radii.append(ts_cutoff)
        elif kind in ("LRCoulomb", "Dipole", "Quadrupole"):
            raise ValueError(f"Source {kind} has no periodic reference here.")
    return max(radii)


def _lattice_translations(
    coord: np.ndarray, cell: np.ndarray, reach: float
) -> np.ndarray:
    """Every translation ``n @ cell`` that can bring an image within ``reach``.

    Axis ``k`` planes are ``1 / |inv(cell)[:, k]|`` apart, so any image within
    ``reach`` has ``|n_k|`` at most ``reach * |inv(cell)[:, k]|`` plus the atoms'
    fractional span along that axis.
    """
    inverse = np.linalg.inv(cell)
    fractional = coord @ inverse
    span = fractional.max(0) - fractional.min(0)
    bound = np.ceil(reach * np.linalg.norm(inverse, axis=0) + span).astype(int)
    axes = [np.arange(-n, n + 1) for n in bound]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid @ cell


def reference_forward(
    declaration: dict[str, Any],
    tensors: dict[str, np.ndarray],
    numbers: np.ndarray,
    coord: np.ndarray,
    charge: float,
    multiplicity: float | None = None,
    *,
    ts_cutoff: float = math.inf,
    cell: np.ndarray | None = None,
    charge_epsilon: float = 1.0e-6,
) -> dict[str, Any]:
    """The source dense forward of one molecule or periodic cell (eV, angstrom, e).

    ``cell`` (rows are lattice vectors, all axes periodic) sums every pair over
    the lattice images ``r_j + n @ cell - r_i`` (``nb_mode`` with ``shifts``),
    excluding only the atom's own untranslated position. ``charge_epsilon`` is
    the source ``ops.nse`` epsilon; it is a parameter only so its effect on the
    equilibrated charges can be isolated.
    """
    kwargs = declaration["kwargs"]
    d2 = kwargs["d2features"]
    channels = kwargs.get("num_charge_channels", 1)
    count = len(numbers)
    rc = float(tensors["aev.rc_s"])
    eta = float(tensors["aev.eta_s"])
    shifts = tensors["aev.shifts_s"]
    if cell is None:
        translations = np.zeros((1, 3))
    else:
        translations = _lattice_translations(
            coord, cell, _periodic_reach(kwargs["outputs"], tensors, rc, ts_cutoff)
        )
    # r_ijn = r_j + t_n - r_i over atoms i, j and lattice translations t_n.
    r = coord[None, :, None, :] + translations[None, None] - coord[:, None, None, :]
    eye = np.eye(count, dtype=bool)[:, :, None] & np.all(translations == 0.0, -1)
    r = np.where(eye[..., None], 1.0, r)
    d = np.linalg.norm(r, axis=-1)
    fc = 0.5 * (np.cos(np.clip(d, 1e-6, rc) * (math.pi / rc)) + 1.0)
    fc = np.where(eye, 0.0, fc)
    gs = np.exp(-eta * (d[..., None] - shifts) ** 2) * fc[..., None]
    u = r / d[..., None]
    gsv = np.concatenate([gs[..., None], gs[..., None] * u[..., None, :]], axis=-1)

    def conv(features: np.ndarray, agh: np.ndarray, two_dimensional: bool) -> np.ndarray:
        if two_dimensional:
            avf = np.einsum("jag,ijngd->iagd", features, gsv)
        else:
            avf = np.einsum("ja,ijngd->iagd", features, gsv)
        scalar = avf[..., 0].reshape(count, -1)
        vector = np.einsum("agh,iagd->iahd", agh, avf[..., 1:])
        return np.concatenate([scalar, (vector**2).sum(-1).reshape(count, -1)], axis=-1)

    a = tensors["afv.weight"][numbers]
    if d2:
        a = a.reshape(count, kwargs["nfeature"], -1)
    if channels == 2:
        half_spin = 0.5 * (_declared(multiplicity, "spin charges") - 1.0)
        target = np.asarray([0.5 * charge + half_spin, 0.5 * charge - half_spin])
    else:
        target = np.asarray([charge])
    q = np.zeros((count, channels))
    passes = 0
    while f"mlps.{passes}.0.weight" in tensors:
        passes += 1
    aim = None
    for index in range(passes):
        flat = a.reshape(count, -1)
        stacked = [flat, conv(a, tensors["conv_a.agh"], d2)]
        if index:
            stacked += [q, conv(q, tensors["conv_q.agh"], False)]
        output = _mlp(tensors, f"mlps.{index}", np.concatenate(stacked, -1), index == 0)
        if index == passes - 1:
            aim = output
            break
        raw_q = output[:, :channels]
        f = output[:, channels : 2 * channels] ** 2
        delta = output[:, 2 * channels :]
        q_u = raw_q if index == 0 else q + raw_q
        flexibility = f.sum(0) + charge_epsilon
        q = q_u + f / flexibility * (target - q_u.sum(0))
        a = a + delta.reshape(a.shape)
    total = q.sum(-1)
    result: dict[str, Any] = {
        "charges": total,
        "residual": target - q.sum(0),
        "aim": aim,
    }
    if channels == 2:
        result["spin_charges"] = q[:, 0] - q[:, 1]
    values: dict[str, np.ndarray] = {}
    energy = None
    atom = None
    for name, entry in kwargs["outputs"].items():
        kind = entry["class"].rsplit(".", 1)[-1]
        options = entry["kwargs"]
        if kind == "Output":
            mlp = options["mlp"]
            value = _mlp(
                tensors,
                f"outputs.{name}.mlp",
                _declared(aim, name),
                mlp.get("last_linear", True),
                _activation(
                    mlp.get("activation_fn", "torch.nn.GELU"),
                    mlp.get("activation_kwargs") or {},
                ),
            )
            if options["key_out"] == "energy":
                atom = value[:, 0]
            else:
                values[options["key_out"]] = value
        elif kind == "AtomicShift":
            atom = (
                _declared(atom, name)
                + tensors[f"outputs.{name}.shifts.weight"][numbers, 0]
            )
        elif kind == "AtomicSum":
            energy = _declared(atom, name).sum()
        elif kind in ("SRCoulomb", "LRCoulomb"):
            radius = float(tensors[f"outputs.{name}.rc"])
            if options.get("envelope", "exp") == "exp":
                ratio = np.clip(d / radius, 0.0, 1.0 - 1e-6)
                envelope = np.exp(-1.0 / (1.0 - ratio**2)) / 0.36787944117144233
            else:
                envelope = 0.5 * (np.cos(np.clip(d, 1e-6, radius) * math.pi / radius) + 1)
            products = (total[:, None] * total[None, :])[..., None]
            pair = np.where(eye, 0.0, envelope * products / d)
            short = 0.5 * HARTREE * BOHR * pair.sum()
            if kind == "SRCoulomb":
                energy = _declared(energy, name) - short
            else:
                # Embedded LRCoulomb(simple, subtract_sr): full minus short range.
                full = np.where(eye, 0.0, products / d)
                energy = (
                    _declared(energy, name) + 0.5 * HARTREE * BOHR * full.sum() - short
                )
        elif kind == "SRRep":
            p = tensors[f"outputs.{name}.params.weight"][numbers]
            alpha = (p[:, None, 0] * p[None, :, 0])[..., None]
            zeff = (p[:, None, 1] * p[None, :, 1])[..., None]
            radius = float(tensors[f"outputs.{name}.rc"])
            ratio = np.clip(d / radius, 0.0, 1.0 - 1e-6)
            cut = np.exp(-1.0 / (1.0 - ratio**2)) / 0.36787944117144233
            pair = np.where(eye, 0.0, np.exp(-alpha * d**1.5) * zeff / d * cut)
            energy = _declared(energy, name) + pair.sum()
        elif kind in ("Dipole", "Quadrupole"):
            position = coord
            if options.get("center_coord", False):
                mass = tensors[f"outputs.{name}.mass"][numbers]
                position = coord - (mass[:, None] * coord).sum(0) / mass.sum()
            if kind == "Dipole":
                result["dipole"] = (total[:, None] * position).sum(0)
            else:
                product = np.concatenate(
                    [position**2, position * np.roll(position, -1, -1)], -1
                )
                raw = (total[:, None] * product).sum(0)
                result["quadrupole"] = np.concatenate([raw[:3] - raw[:3].mean(), raw[3:]])
        elif kind == "DispParam":
            reference = tensors[f"outputs.{name}.disp_param0"][numbers]
            values[options["key_out"]] = reference * np.exp(
                np.clip(values[options["key_in"]], -4.0, 4.0)
            )
            result["coefficients"] = values[options["key_out"]]
        elif kind == "D3TS":
            c6, alpha = values[options.get("key_in", "disp_param")].T
            denominator = (
                c6[:, None] * alpha[None, :] / alpha[:, None]
                + c6[None, :] * alpha[:, None] / alpha[None, :]
            )
            c6ij = 2 * c6[:, None] * c6[None, :] / np.maximum(denominator, 1e-4)
            rr = tensors[f"outputs.{name}.r4r2"][numbers]
            rrij = (3 * rr[:, None] * rr[None, :])[..., None]
            r0 = options["a1"] * np.sqrt(rrij) + options["a2"]
            bohr = d / BOHR
            pair = c6ij[..., None] * (
                options.get("s6", 1.0) / (bohr**6 + r0**6)
                + options["s8"] * rrij / (bohr**8 + r0**8)
            )
            pair = np.where(eye | (d >= ts_cutoff), 0.0, pair)
            energy = _declared(energy, name) - 0.5 * HARTREE * pair.sum()
    result["energy"] = energy
    return result
