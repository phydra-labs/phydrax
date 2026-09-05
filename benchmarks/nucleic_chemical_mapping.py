"""Actual CC0 RMDB processed-observation fit and held-out mutant prediction.

Depositor structures are designed hypotheses, not solved structures; the affine
accessibility law is evaluated, not asserted accurate. Marginal supplied errors
are treated as SD under an explicit diagonal-noise approximation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

from phydrax.applications.nucleic_acid_biophysics.observations import (
    AccessibilityReactivityModel,
    import_processed_rdat,
)
from phydrax.qualification import ReferenceArtifactManifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests/fixtures/nucleic_acid_biophysics/TODEX_DMS_0000.rdat",
    )
    parser.add_argument("--training-constructs", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = args.fixture.read_bytes()
    record = json.loads(args.fixture.with_suffix(".source.json").read_text())
    source = ReferenceArtifactManifest(
        "RMDB:TODEX_DMS_0000",
        checksum_algorithm="sha256",
        checksum=record["sha256"],
        size_bytes=record["size_bytes"],
        license_id=record["license_id"],
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="CC0-database-content",
        nondimensionalization={"normalized-reactivity": 1.0},
        uncertainty=None,
        lineage_ids=(record["source_url"],),
    )
    imported = import_processed_rdat(
        payload,
        source,
        requested_use={"training_use": True},
        error_semantics="standard-deviation",
    )
    if not 1 <= args.training_constructs < len(imported.entries) or args.repeats < 1:
        raise ValueError("The benchmark must withhold at least one complete construct.")

    def model(entries):
        signals = tuple(
            np.asarray(
                [
                    float(entry.declared_structure[key.position] == ".")
                    for key in entry.observation.nucleotide_keys
                ]
            )
            for entry in entries
        )
        return AccessibilityReactivityModel(
            tuple(entry.observation for entry in entries),
            signals,
            baseline_groups=("shared",) * len(entries),
            condition_features=np.zeros((len(entries), 0)),
        )

    training = model(imported.entries[: args.training_constructs])
    withheld = model(imported.entries[args.training_constructs :])
    fitted, fit_seconds = measure_synchronized(training.fit)
    parameters = fitted.optimization.parameters

    def held_scores(values):
        predictions = withheld.predict(values)
        return jnp.stack(
            tuple(
                obs.score(pred).log_probability
                for obs, pred in zip(withheld.observations, predictions, strict=True)
            )
        )

    compiled = jax.jit(held_scores)
    executable, compile_timing = measure_lower_and_compile(
        lambda: compiled.lower(parameters), lambda lowered: lowered.compile()
    )
    scores, timing = measure_repeated(
        lambda: executable(parameters), warmup=1, repeats=args.repeats
    )
    predictions = withheld.predict(parameters)
    residual = jnp.concatenate(
        tuple(
            pred - obs.reactivity
            for pred, obs in zip(predictions, withheld.observations, strict=True)
        )
    )
    standardized = jnp.concatenate(
        tuple(
            obs.residual(pred)
            for pred, obs in zip(predictions, withheld.observations, strict=True)
        )
    )
    result = {
        "environment": capture_environment().to_dict(),
        "source_url": record["source_url"],
        "source_sha256": source.checksum,
        "license": source.license_id,
        "training_constructs": args.training_constructs,
        "withheld_constructs": len(withheld.observations),
        "withheld_measurements": int(residual.size),
        "negative_reactivities_preserved": sum(
            int(jnp.sum(entry.observation.reactivity < 0)) for entry in imported.entries
        ),
        "fit_total_seconds_including_native_preparation_and_compilation": fit_seconds,
        "score_compile": asdict(compile_timing),
        "score_execution_seconds": timing.to_seconds_dict(),
        "optimizer_successful": bool(fitted.optimization.successful),
        "parameter_identifiable": bool(fitted.identifiable),
        "design_rank": int(fitted.design_rank),
        "withheld_rmse": float(jnp.sqrt(jnp.mean(residual**2))),
        "withheld_chi_square_per_observation": float(jnp.mean(standardized**2)),
        "withheld_log_likelihood": np.asarray(scores).tolist(),
        "uncertainty": "supplied REACTIVITY_ERROR treated as SD; no measured cross-position covariance",
        "structural_reference": "depositor-designed hypothesis, not experimentally solved pairing",
        "claim": "actual fit and withheld predictive score; no preasserted experimental accuracy",
    }
    encoded = json.dumps(result, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
