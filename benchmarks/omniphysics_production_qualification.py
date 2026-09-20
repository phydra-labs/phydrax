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
    content = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(content, encoding="utf-8")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
