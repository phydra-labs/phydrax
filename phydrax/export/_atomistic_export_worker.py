#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Isolated trace-and-compile worker of the frozen atomistic IREE export.

``save_atomistic_iree`` runs this module as ``__main__`` through
``run_pinned_command`` in the parent's own interpreter, started with ``-I -S``,
an environment holding only ``EQX_ON_ERROR=nan``, ``JAX_ENABLE_X64`` and
``JAX_PLATFORMS=cpu``, and the parent's implementation root ahead of the
declared site directories. Arguments::

    <implementation-root> <request-archive>

The working directory is the private run directory. The worker refuses unless
the Equinox ``nan`` policy is active and it imported Phydrax from exactly the
declared root, rebuilds the model, provider plan, structure and contract from
the pickle-free request archive, requires the identical contract and a
callback-free program, compiles it, and writes ``module.vmfb``,
``manifest.json`` and ``result.json``. A refusal exits with status 3 and a
``REFUSED:`` diagnostic line. Nothing here verifies numerical parity: the
parent checks the returned module before publishing it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path


_REFUSED_EXIT = 3


def _export(root: Path, request_path: Path, output: Path, /) -> None:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    import phydrax

    from . import _atomistic as atomistic
    from ._iree import save_iree

    refusal = atomistic.AtomisticExportRefusal
    probe = jax.jit(lambda flag: eqx.error_if(jnp.zeros(()), flag, "policy probe"))
    try:
        poisoned = bool(jnp.isnan(probe(jnp.asarray(True))))
    except Exception as error:
        raise refusal(
            "The export worker requires the Equinox nan error policy."
        ) from error
    if not poisoned:
        raise refusal("The export worker requires the Equinox nan error policy.")
    imported = Path(phydrax.__file__).resolve(strict=True).parents[1]
    if (
        imported != root.resolve(strict=True)
        or atomistic._implementation_root() != imported
    ):
        raise refusal("The export worker imported Phydrax from another source tree.")
    request = atomistic._read_export_request(request_path)
    runtime = atomistic._runtime_record()
    if request.runtime != runtime:
        raise refusal("The export worker runtime differs from the parent runtime.")
    provider, native = atomistic._native_provider(
        request.plan, request.structure, request.units
    )
    contract = atomistic.prepare_atomistic_iree_contract(
        provider, native, route=request.contract.route
    )
    if contract != request.contract:
        raise refusal("The worker rebuild differs from the parent export contract.")
    inputs = contract.pack_inputs(native)
    forward = atomistic._exported_evaluation(provider, native, contract)
    callbacks = [
        site
        for site in atomistic._runtime_guard_sites(forward, inputs)
        if site.kind == "host-callback"
    ]
    if callbacks:
        raise refusal(
            "The frozen program keeps host callbacks under the nan policy: "
            + "; ".join(site.location for site in callbacks[:8])
        )
    exported = save_iree(
        forward,
        output / "staged",
        inputs=inputs,
        input_names=contract.input_names,
        output_names=contract.output_names,
        policy=request.policy,
        key=None,
        validate=False,
        domain_contract=contract.to_dict(),
    )
    manifest = exported.manifest
    staged = output / "staged"
    (output / atomistic._MODULE_NAME).write_bytes(
        (staged / manifest.module_file).read_bytes()
    )
    (output / atomistic._MANIFEST_NAME).write_bytes(
        (staged / "manifest.json").read_bytes()
    )
    (output / atomistic._RESULT_NAME).write_text(
        json.dumps(
            {
                "format": atomistic._RESULT_FORMAT,
                "contract_id": contract.contract_id,
                "module_sha256": manifest.module_sha256,
                "implementation_root": str(imported),
                "runtime": runtime,
            },
            allow_nan=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main(arguments: Sequence[str], /) -> int:
    if len(arguments) != 2:
        print(
            "usage: _atomistic_export_worker <implementation-root> <request-archive>",
            file=sys.stderr,
        )
        return 2
    from ._atomistic import AtomisticExportRefusal

    try:
        _export(Path(arguments[0]), Path(arguments[1]), Path.cwd())
    except AtomisticExportRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return _REFUSED_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
