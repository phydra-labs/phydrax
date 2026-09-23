import argparse
import json
from pathlib import Path

from phydrax.qualification import builtin_omniphysics_qualification_evidence


def run():
    return builtin_omniphysics_qualification_evidence().to_record()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = run()
    if args.output is not None:
        from benchmarks._io import write_json_atomic

        write_json_atomic(args.output, record)
    print(json.dumps(record, allow_nan=False, sort_keys=True))
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
