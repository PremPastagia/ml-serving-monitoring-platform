#!/usr/bin/env python
"""Download the UCI Adult dataset and verify it against the pinned checksums.

    python scripts/fetch_data.py            # download anything missing, then verify
    python scripts/fetch_data.py --force    # re-download even if present
    python scripts/fetch_data.py --verify   # verify only; never touch the network

Verification is not optional: the dataset version id is derived from these bytes, so
an altered or truncated file would silently invalidate every recorded metric.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from mlserve.config import load_config
from mlserve.data.ingest import SOURCES, ChecksumMismatch, sha256_file, verify_raw_files

TIMEOUT_SECONDS = 120
RETRIES = 3


def download(url: str, destination: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
                destination.write_bytes(response.read())
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            print(f"  attempt {attempt}/{RETRIES} failed: {exc}")
    raise RuntimeError(f"could not download {url}: {last_error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--verify", action="store_true", help="verify only, no network access")
    parser.add_argument("--raw-dir", default=None, help="override the configured raw directory")
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir) if args.raw_dir else load_config().path("paths.raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.verify:
        for name, meta in SOURCES.items():
            destination = raw_dir / name
            if destination.exists() and not args.force:
                if sha256_file(destination) == meta["sha256"]:
                    print(f"present  {name}  (checksum matches)")
                    continue
                print(f"stale    {name}  (checksum differs; re-downloading)")
            print(f"fetching {name} from {meta['url']}")
            download(meta["url"], destination)

    try:
        observed = verify_raw_files(raw_dir, strict=True)
    except (ChecksumMismatch, FileNotFoundError) as exc:
        print(f"\nVERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("\nchecksums verified:")
    for name, digest in observed.items():
        print(f"  {name:12s} {digest}  ({SOURCES[name]['role']})")
    print("FETCH_DATA_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
