"""Generate tiny MACE source-provider reference fixtures.

Run with an environment providing mace-torch 0.3.16 and e3nn 0.4.4 (not a
Phydrax dependency), for example::

    .tmp/mace-provider/.venv/bin/python tests/fixtures/mace_torch_0_3_16/generate.py

Each case builds a seeded, randomly perturbed float64 source model, evaluates
energies, forces, and virials of a finite molecule and a periodic triclinic
crystal with the provider, and stores the provider's complete state dictionary,
its real Wigner 3j tables, and sampled source harmonics. The fixtures are an
independent oracle for native reconstruction; they are not checkpoints of any
published model.
"""

from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


_DIRECTORY = Path(__file__).resolve().parent
_GATE_SCALE_KEY = "normalize2mom_silu"

_CASES: dict[str, dict[str, Any]] = {
    "one_residual": {
        "class": "ScaleShiftMACE",
        "first": "RealAgnosticResidualInteractionBlock",
        "rest": "RealAgnosticResidualInteractionBlock",
        "interactions": 1,
        "correlation": [3],
        "heads": ["Default"],
        "pair_repulsion": False,
        "distance_transform": "None",
        "apply_cutoff": True,
    },
    "two_scale_shift": {
        "class": "ScaleShiftMACE",
        "first": "RealAgnosticInteractionBlock",
        "rest": "RealAgnosticResidualInteractionBlock",
        "interactions": 2,
        "correlation": [3, 2],
        "heads": ["Default"],
        "pair_repulsion": True,
        "distance_transform": "Agnesi",
        "apply_cutoff": True,
    },
    "three_density": {
        "class": "ScaleShiftMACE",
        "first": "RealAgnosticDensityInteractionBlock",
        "rest": "RealAgnosticDensityResidualInteractionBlock",
        "interactions": 3,
        "correlation": [3, 2, 2],
        "heads": ["Default"],
        "pair_repulsion": True,
        "distance_transform": "Agnesi",
        "apply_cutoff": True,
    },
    "two_multihead_unscaled": {
        "class": "MACE",
        "first": "RealAgnosticResidualInteractionBlock",
        "rest": "RealAgnosticResidualInteractionBlock",
        "interactions": 2,
        "correlation": [2, 2],
        "heads": ["first", "second"],
        "pair_repulsion": True,
        "distance_transform": "None",
        "apply_cutoff": False,
    },
}

_STRUCTURES: dict[str, dict[str, Any]] = {
    "molecule": {
        "symbols": "OHH",
        "positions": [[0.0, 0.0, 0.0], [0.95, 0.1, 0.0], [-0.25, 0.85, 0.2]],
        "cell": None,
    },
    "crystal": {
        "symbols": "OH",
        "positions": [[0.1, 0.2, 0.05], [1.05, 0.65, 0.9]],
        "cell": [[2.4, 0.0, 0.0], [0.6, 2.2, 0.0], [0.3, 0.4, 2.6]],
    },
}


def _evaluate(
    modules: dict[str, Any], model: Any, structure: dict[str, Any], head: int, /
) -> dict[str, np.ndarray]:
    torch, ase, data, tools = (
        modules["torch"],
        modules["ase"],
        modules["data"],
        modules["tools"],
    )
    periodic = structure["cell"] is not None
    atoms = ase.Atoms(
        structure["symbols"],
        positions=structure["positions"],
        cell=structure["cell"],
        pbc=periodic,
    )
    table = tools.AtomicNumberTable([1, 8])
    config = data.config_from_atoms(atoms)
    batch = data.AtomicData.from_config(config, z_table=table, cutoff=3.0).to_dict()
    batch["batch"] = torch.zeros(len(atoms), dtype=torch.long)
    batch["ptr"] = torch.tensor([0, len(atoms)])
    batch["head"] = torch.tensor([head], dtype=torch.long)
    output = model(batch, compute_force=True, compute_virials=periodic, training=True)
    names, parameters = zip(*model.named_parameters())
    gradients = torch.autograd.grad(output["energy"].sum(), parameters, retain_graph=True)
    result = {
        "energy": output["energy"].detach().numpy(),
        "forces": output["forces"].detach().numpy(),
        **{
            f"parameter_gradient/{name}": gradient.detach().numpy()
            for name, gradient in zip(names, gradients)
        },
    }
    if periodic:
        result["virials"] = output["virials"].detach().numpy()
    return result


def _case(modules: dict[str, Any], name: str, case: dict[str, Any], /) -> dict[str, Any]:
    torch, o3, mace = modules["torch"], modules["o3"], modules["mace"]
    torch.manual_seed(20261003)
    heads = case["heads"]
    arguments: dict[str, Any] = {
        "r_max": 3.0,
        "num_bessel": 4,
        "num_polynomial_cutoff": 5,
        "max_ell": 2,
        "interaction_cls": mace.interaction_classes[case["rest"]],
        "interaction_cls_first": mace.interaction_classes[case["first"]],
        "num_interactions": case["interactions"],
        "num_elements": 2,
        "hidden_irreps": o3.Irreps("4x0e+4x1o"),
        "MLP_irreps": o3.Irreps("5x0e"),
        "gate": torch.nn.functional.silu,
        "atomic_energies": np.asarray([[-1.0, -2.0], [-1.5, -2.5]][: len(heads)]),
        "avg_num_neighbors": 2.0,
        "atomic_numbers": [1, 8],
        "correlation": case["correlation"],
        "radial_MLP": [8],
        "pair_repulsion": case["pair_repulsion"],
        "distance_transform": case["distance_transform"],
        "apply_cutoff": case["apply_cutoff"],
        "heads": heads,
    }
    if case["class"] == "ScaleShiftMACE":
        model = mace.ScaleShiftMACE(
            atomic_inter_scale=1.3, atomic_inter_shift=0.2, **arguments
        )
    else:
        model = mace.MACE(**arguments)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.3 * torch.randn_like(parameter))
    arrays: dict[str, np.ndarray] = {
        f"state/{key}": value.detach().numpy()
        for key, value in model.state_dict().items()
    }
    for left in range(3):
        for right in range(3):
            for output in range(abs(left - right), min(left + right, 2) + 1):
                arrays[f"w3j/{left}_{right}_{output}"] = o3.wigner_3j(
                    left, right, output
                ).numpy()
    vectors = np.random.default_rng(7).normal(size=(64, 3))
    arrays["harmonics/vectors"] = vectors
    arrays["harmonics/values"] = o3.spherical_harmonics(
        o3.Irreps.spherical_harmonics(2),
        torch.tensor(vectors),
        normalize=True,
        normalization="component",
    ).numpy()
    for structure_name, structure in _STRUCTURES.items():
        arrays[f"{structure_name}/positions"] = np.asarray(structure["positions"])
        if structure["cell"] is not None:
            arrays[f"{structure_name}/cell"] = np.asarray(structure["cell"])
        for index, head in enumerate(heads):
            for key, value in _evaluate(modules, model, structure, index).items():
                arrays[f"{structure_name}/{head}/{key}"] = value
    # The compressed npz layout of numpy.savez_compressed, written member by
    # member so every array is stored without pickling.
    with zipfile.ZipFile(
        _DIRECTORY / f"{name}.npz", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for key, value in arrays.items():
            with archive.open(f"{key}.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(
                    member, np.asanyarray(value), allow_pickle=False
                )
    activation = model.interactions[0].conv_tp_weights
    gate_scale = float(next(iter(activation._modules.values())).act.cst)
    return {**case, _GATE_SCALE_KEY: gate_scale}


def main() -> None:
    torch = importlib.import_module("torch")
    # e3nn 0.4.4 loads its packaged constants with a slice global under the
    # weights-only loader of recent torch releases.
    torch.serialization.add_safe_globals([slice])
    torch.set_default_dtype(torch.float64)
    modules = {
        "torch": torch,
        "ase": importlib.import_module("ase"),
        "o3": importlib.import_module("e3nn.o3"),
        "mace": importlib.import_module("mace.modules"),
        "data": importlib.import_module("mace.data"),
        "tools": importlib.import_module("mace.tools"),
    }
    metadata = importlib.import_module("importlib.metadata")
    manifest = {
        "provider": {
            name: metadata.version(name)
            for name in ("mace-torch", "e3nn", "torch", "ase")
        },
        "generator": "tests/fixtures/mace_torch_0_3_16/generate.py",
        "units": {"length": "angstrom", "energy": "eV"},
        "virial_convention": "provider virials = -dE/d(symmetric strain)",
        "structures": _STRUCTURES,
        "cases": {name: _case(modules, name, case) for name, case in _CASES.items()},
    }
    (_DIRECTORY / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
