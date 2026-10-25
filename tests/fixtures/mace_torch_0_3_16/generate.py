"""Generate tiny MACE source-provider reference fixtures.

Run with an environment providing mace-torch 0.3.16 and e3nn 0.4.4 (and the
pinned LES 0.2.0 for the MACELES case; none is a Phydrax dependency), for
example::

    .tmp/mace-provider/.venv/bin/python tests/fixtures/mace_torch_0_3_16/generate.py

Each case builds a seeded, randomly perturbed float64 source model, evaluates
energies, forces, and virials of a finite molecule and a periodic triclinic
crystal with the provider, and stores the provider's complete state dictionary,
its real Wigner 3j tables, and sampled source harmonics. The manifest records
each case's declaration and the provider's realized normalization constants
(``normalize2mom_silu`` of the radial networks, the readout activation, and
the nonlinear-block gate constants). The fixtures are an independent oracle
for native reconstruction; they are not checkpoints of any published model.

Family cases cover the attention-residual and residual-nonlinear interactions
with a nonlinear-bias readout, Gaussian/Chebyshev bases with Soft/Agnesi
transforms, the SO(3) parity option, first-layer edge irreps, kept last-layer
irreps, the charge/spin joint embedding with its embedding readout, and
MACELES on both the finite and the periodic structure.

``tests/fixtures/mace_torch_0_3_15/generate.py`` reuses ``generate`` for the
0.3.15 release cases.
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
_SEED = 20261003

_LES_ARGUMENTS: dict[str, Any] = {
    "n_layers": 3,
    "n_hidden": [32, 16],
    "add_linear_nn": True,
    "output_scaling_factor": 0.1,
    "sigma": 1.0,
    "dl": 2.0,
    "remove_self_interaction": True,
    "is_periodic": None,
    "N_max": 10,
    "remove_mean": True,
    "epsilon_factor": 1.0,
    "use_atomwise": False,
    "use_fixed_atomic_charges": False,
    "fixed_atomic_charges_scaling_factor": 0.5,
    "use_atomic_alpha": False,
    "use_epsilon_r_scaling": False,
    "compute_bec": False,
    "bec_output_index": None,
}

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

# One seeded tiny case per admitted family (shared fields as above).
_FAMILY_SHARED: dict[str, Any] = {
    "class": "ScaleShiftMACE",
    "first": "RealAgnosticInteractionBlock",
    "rest": "RealAgnosticResidualInteractionBlock",
    "interactions": 2,
    "correlation": [2, 2],
    "heads": ["Default"],
    "pair_repulsion": False,
    "distance_transform": "None",
    "apply_cutoff": True,
}
FAMILY_CASES: dict[str, dict[str, Any]] = {
    "two_attention": {
        **_FAMILY_SHARED,
        "first": "RealAgnosticAttResidualInteractionBlock",
        "rest": "RealAgnosticAttResidualInteractionBlock",
    },
    "two_nonlinear_bias": {
        **_FAMILY_SHARED,
        "first": "RealAgnosticResidualNonLinearInteractionBlock",
        "rest": "RealAgnosticResidualNonLinearInteractionBlock",
        "readout_class": "NonLinearBiasReadoutBlock",
    },
    "two_gaussian_soft": {
        **_FAMILY_SHARED,
        "radial_type": "gaussian",
        "distance_transform": "Soft",
    },
    "two_chebyshev_agnesi": {
        **_FAMILY_SHARED,
        "radial_type": "chebyshev",
        "distance_transform": "Agnesi",
    },
    "two_so3": {**_FAMILY_SHARED, "use_so3": True, "hidden_irreps": "4x0e+4x1e"},
    "two_edge_irreps_first": {
        **_FAMILY_SHARED,
        "edge_irreps": "3x0e+3x1o",
        "use_edge_irreps_first": True,
    },
    "two_keep_last_layer": {**_FAMILY_SHARED, "keep_last_layer_irreps": True},
    "two_joint_embedding": {
        **_FAMILY_SHARED,
        "embedding_specs": {
            "total_charge": {
                "type": "categorical",
                "per": "graph",
                "num_classes": 5,
                "emb_dim": 3,
                "offset": 2,
            },
            "total_spin": {
                "type": "categorical",
                "per": "graph",
                "num_classes": 4,
                "emb_dim": 2,
                "offset": 0,
            },
        },
        "use_embedding_readout": True,
        "conditions": {"total_charge": 1, "total_spin": 2},
    },
    "two_maceles": {
        **_FAMILY_SHARED,
        "class": "MACELES",
        "les_arguments": _LES_ARGUMENTS,
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
    modules: dict[str, Any],
    model: Any,
    structure: dict[str, Any],
    head: int,
    conditions: dict[str, int],
    /,
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
    for name, value in conditions.items():
        config.properties[name] = float(value)
    batch = data.AtomicData.from_config(config, z_table=table, cutoff=3.0).to_dict()
    batch["batch"] = torch.zeros(len(atoms), dtype=torch.long)
    batch["ptr"] = torch.tensor([0, len(atoms)])
    batch["head"] = torch.tensor([head], dtype=torch.long)
    # One unbatched graph: graph-level charge/spin become one-graph vectors.
    for name in ("total_charge", "total_spin"):
        if batch.get(name) is not None and batch[name].ndim == 0:
            batch[name] = batch[name].reshape(1)
    output = model(batch, compute_force=True, compute_virials=periodic, training=True)
    names, parameters = zip(*model.named_parameters())
    gradients = torch.autograd.grad(
        output["energy"].sum(), parameters, retain_graph=True, allow_unused=True
    )
    result = {
        "energy": output["energy"].detach().numpy(),
        "forces": output["forces"].detach().numpy(),
        **{
            f"parameter_gradient/{name}": (
                torch.zeros_like(parameter) if gradient is None else gradient
            )
            .detach()
            .numpy()
            for name, parameter, gradient in zip(names, parameters, gradients)
        },
    }
    if periodic:
        result["virials"] = output["virials"].detach().numpy()
    return result


def _constants(model: Any, /) -> dict[str, Any]:
    """The provider's realized normalization constants of one model."""

    constants: dict[str, Any] = {_GATE_SCALE_KEY: None}
    for interaction in model.interactions:
        network = interaction.conv_tp_weights
        if hasattr(network, "layer0"):
            constants[_GATE_SCALE_KEY] = float(network.layer0.act.cst)
            break
    readout = model.readouts[-1]
    if hasattr(readout, "non_linearity"):
        constants["readout_scale"] = float(readout.non_linearity.acts[0].cst)
    for interaction in model.interactions:
        gate = getattr(interaction, "equivariant_nonlin", None)
        if gate is None:
            continue
        if hasattr(gate, "_scalar_cst"):
            constants["gate_scales"] = [float(gate._scalar_cst), float(gate._gate_cst)]
        else:
            constants["gate_scales"] = [
                float(gate.act_scalars.acts[0].cst),
                float(gate.act_gates.acts[0].cst),
            ]
        break
    return constants


def _case(
    modules: dict[str, Any], directory: Path, name: str, case: dict[str, Any], /
) -> dict[str, Any]:
    torch, o3, mace = modules["torch"], modules["o3"], modules["mace"]
    torch.manual_seed(_SEED)
    heads = case["heads"]
    classes = mace.interaction_classes
    arguments: dict[str, Any] = {
        "r_max": 3.0,
        "num_bessel": 4,
        "num_polynomial_cutoff": 5,
        "max_ell": 2,
        "interaction_cls": classes[case["rest"]],
        "interaction_cls_first": classes[case["first"]],
        "num_interactions": case["interactions"],
        "num_elements": 2,
        "hidden_irreps": o3.Irreps(case.get("hidden_irreps", "4x0e+4x1o")),
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
    # Family arguments are passed only when declared, so the four original
    # cases construct exactly as before.
    for key in (
        "radial_type",
        "use_so3",
        "use_edge_irreps_first",
        "use_embedding_readout",
    ):
        if key in case:
            arguments[key] = case[key]
    if "edge_irreps" in case:
        arguments["edge_irreps"] = o3.Irreps(case["edge_irreps"])
    if "readout_class" in case:
        arguments["readout_cls"] = getattr(modules["blocks"], case["readout_class"])
    if "keep_last_layer_irreps" in case:
        arguments["keep_last_layer_irreps"] = case["keep_last_layer_irreps"]
    if "embedding_specs" in case:
        arguments["embedding_specs"] = {
            key: dict(value) for key, value in case["embedding_specs"].items()
        }
    match case["class"]:
        case "ScaleShiftMACE":
            model = mace.ScaleShiftMACE(
                atomic_inter_scale=1.3, atomic_inter_shift=0.2, **arguments
            )
        case "MACELES":
            extensions = importlib.import_module("mace.modules.extensions")
            model = extensions.MACELES(
                les_arguments=dict(case["les_arguments"]),
                atomic_inter_scale=1.3,
                atomic_inter_shift=0.2,
                **arguments,
            )
        case _:
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
    conditions = dict(case.get("conditions", {}))
    for structure_name, structure in _STRUCTURES.items():
        arrays[f"{structure_name}/positions"] = np.asarray(structure["positions"])
        if structure["cell"] is not None:
            arrays[f"{structure_name}/cell"] = np.asarray(structure["cell"])
        for index, head in enumerate(heads):
            for key, value in _evaluate(
                modules, model, structure, index, conditions
            ).items():
                arrays[f"{structure_name}/{head}/{key}"] = value
    # The compressed npz layout of numpy.savez_compressed, written member by
    # member so every array is stored without pickling.
    with zipfile.ZipFile(
        directory / f"{name}.npz", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for key, value in arrays.items():
            with archive.open(f"{key}.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(
                    member, np.asanyarray(value), allow_pickle=False
                )
    return {**case, **_constants(model)}


def generate(directory: Path, cases: dict[str, dict[str, Any]], release: str, /) -> None:
    """Write every case and the manifest of one provider release into ``directory``."""

    torch = importlib.import_module("torch")
    # e3nn 0.4.4 loads its packaged constants with a slice global under the
    # weights-only loader of recent torch releases.
    torch.serialization.add_safe_globals([slice])
    torch.set_default_dtype(torch.float64)
    metadata = importlib.import_module("importlib.metadata")
    installed = metadata.version("mace-torch")
    if installed != release:
        raise SystemExit(f"These fixtures need mace-torch {release}, not {installed}.")
    modules = {
        "torch": torch,
        "ase": importlib.import_module("ase"),
        "o3": importlib.import_module("e3nn.o3"),
        "mace": importlib.import_module("mace.modules"),
        "blocks": importlib.import_module("mace.modules.blocks"),
        "data": importlib.import_module("mace.data"),
        "tools": importlib.import_module("mace.tools"),
    }
    distributions = ["mace-torch", "e3nn", "torch", "ase"]
    if any(case["class"] == "MACELES" for case in cases.values()):
        distributions.append("les")
    manifest = {
        "provider": {name: metadata.version(name) for name in distributions},
        "generator": f"tests/fixtures/{directory.name}/generate.py",
        "units": {"length": "angstrom", "energy": "eV"},
        "virial_convention": "provider virials = -dE/d(symmetric strain)",
        "structures": _STRUCTURES,
        "cases": {
            name: _case(modules, directory, name, case) for name, case in cases.items()
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    generate(_DIRECTORY, {**_CASES, **FAMILY_CASES}, "0.3.16")


if __name__ == "__main__":
    main()
