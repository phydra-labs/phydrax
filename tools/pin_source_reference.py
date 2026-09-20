#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--licence", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--concept", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = {
        "source_id": args.source_id,
        "repository_url": args.url,
        "revision": args.revision,
        "archive_digest": digest(args.archive),
        "licence_digest": digest(args.licence),
        "concepts": sorted(args.concept),
        "review_status": "unreviewed",
    }
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
