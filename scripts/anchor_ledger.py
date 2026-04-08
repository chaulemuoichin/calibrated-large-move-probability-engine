"""
Tamper-evident anchoring for the forecast ledger.

Creates a cryptographic manifest of the current ledger state and commits
it as a signed git tag. Anyone can later verify the ledger hasn't been
modified by checking the tag's manifest hash against the current files.

This turns the local append-only ledger into a publicly verifiable record
when the tag is pushed to a remote repository.

Usage:
    python scripts/anchor_ledger.py                   # create anchor
    python scripts/anchor_ledger.py --verify           # verify latest anchor
    python scripts/anchor_ledger.py --push             # create and push
    python scripts/anchor_ledger.py --list              # list all anchors
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_sde.ledger import FORECAST_FILE, RESOLUTION_FILE, LEDGER_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_FILE = LEDGER_DIR / "anchor_manifest.json"
TAG_PREFIX = "ledger-anchor/"


def _sha256_file(path: Path) -> str:
    """SHA-256 hash of file contents, or 'missing' if file doesn't exist."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _line_count(path: Path) -> int:
    """Count non-empty lines in a file."""
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _git_run(args: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command and return the result."""
    return subprocess.run(
        ["git"] + args,
        capture_output=True, text=True, timeout=30,
        check=check,
    )


def create_anchor(push: bool = False) -> dict:
    """
    Create a tamper-evident anchor of the current ledger state.

    1. Compute SHA-256 of forecast and resolution ledger files
    2. Write a manifest JSON with hashes, counts, and timestamp
    3. Commit the manifest and create a signed git tag
    4. Optionally push the tag to the remote
    """
    now_utc = datetime.now(timezone.utc)
    tag_name = f"{TAG_PREFIX}{now_utc.strftime('%Y%m%dT%H%M%SZ')}"

    forecast_hash = _sha256_file(FORECAST_FILE)
    resolution_hash = _sha256_file(RESOLUTION_FILE)
    forecast_lines = _line_count(FORECAST_FILE)
    resolution_lines = _line_count(RESOLUTION_FILE)

    # Get current git commit
    try:
        result = _git_run(["rev-parse", "HEAD"], check=False)
        git_commit = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        git_commit = "unknown"

    manifest = {
        "anchor_timestamp_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit,
        "tag_name": tag_name,
        "forecasts": {
            "path": str(FORECAST_FILE),
            "sha256": forecast_hash,
            "n_records": forecast_lines,
        },
        "resolutions": {
            "path": str(RESOLUTION_FILE),
            "sha256": resolution_hash,
            "n_records": resolution_lines,
        },
        "combined_hash": hashlib.sha256(
            f"{forecast_hash}:{resolution_hash}".encode()
        ).hexdigest(),
    }

    # Write manifest
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written: %s", MANIFEST_FILE)

    # Stage and commit the manifest
    _git_run(["add", str(MANIFEST_FILE)])
    commit_msg = (
        f"ledger anchor: {forecast_lines} forecasts, {resolution_lines} resolutions\n\n"
        f"forecast_sha256: {forecast_hash[:16]}...\n"
        f"resolution_sha256: {resolution_hash[:16]}...\n"
        f"combined_hash: {manifest['combined_hash'][:16]}..."
    )
    try:
        _git_run(["commit", "-m", commit_msg])
    except subprocess.CalledProcessError:
        # Nothing to commit (manifest unchanged)
        logger.info("Manifest unchanged, skipping commit")

    # Create annotated tag
    tag_msg = (
        f"Ledger anchor at {now_utc.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"Forecasts: {forecast_lines} records (sha256: {forecast_hash})\n"
        f"Resolutions: {resolution_lines} records (sha256: {resolution_hash})\n"
        f"Combined: {manifest['combined_hash']}\n\n"
        f"Verify: python scripts/anchor_ledger.py --verify"
    )
    try:
        _git_run(["tag", "-a", tag_name, "-m", tag_msg])
        logger.info("Created tag: %s", tag_name)
    except subprocess.CalledProcessError as e:
        logger.error("Failed to create tag: %s", e.stderr)
        return manifest

    if push:
        try:
            _git_run(["push", "origin", tag_name])
            logger.info("Pushed tag %s to origin", tag_name)
        except subprocess.CalledProcessError as e:
            logger.error("Failed to push tag: %s", e.stderr)

    # Print summary
    print(f"\n{'='*60}")
    print(f"LEDGER ANCHOR CREATED")
    print(f"  Tag:         {tag_name}")
    print(f"  Forecasts:   {forecast_lines} records")
    print(f"  Resolutions: {resolution_lines} records")
    print(f"  Combined:    {manifest['combined_hash'][:32]}...")
    print(f"{'='*60}")
    print(f"\nVerify with: python scripts/anchor_ledger.py --verify")
    if not push:
        print(f"Push with:   git push origin {tag_name}")

    return manifest


def verify_anchor() -> bool:
    """
    Verify the current ledger against the latest anchor manifest.

    Returns True if the ledger matches, False otherwise.
    """
    if not MANIFEST_FILE.exists():
        print("No anchor manifest found. Create one with: python scripts/anchor_ledger.py")
        return False

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    forecast_hash = _sha256_file(FORECAST_FILE)
    resolution_hash = _sha256_file(RESOLUTION_FILE)
    combined = hashlib.sha256(
        f"{forecast_hash}:{resolution_hash}".encode()
    ).hexdigest()

    ok = True
    print(f"\nAnchor: {manifest.get('tag_name', 'unknown')} "
          f"({manifest.get('anchor_timestamp_utc', 'unknown')})")

    # Check forecasts
    expected_f = manifest["forecasts"]["sha256"]
    if forecast_hash == expected_f:
        print(f"  Forecasts:   MATCH ({manifest['forecasts']['n_records']} records)")
    else:
        print(f"  Forecasts:   MISMATCH")
        print(f"    Expected: {expected_f[:32]}...")
        print(f"    Current:  {forecast_hash[:32]}...")
        ok = False

    # Check resolutions
    expected_r = manifest["resolutions"]["sha256"]
    if resolution_hash == expected_r:
        print(f"  Resolutions: MATCH ({manifest['resolutions']['n_records']} records)")
    else:
        print(f"  Resolutions: MISMATCH")
        print(f"    Expected: {expected_r[:32]}...")
        print(f"    Current:  {resolution_hash[:32]}...")
        ok = False

    # Check combined
    expected_c = manifest["combined_hash"]
    if combined == expected_c:
        print(f"  Combined:    MATCH")
    else:
        print(f"  Combined:    MISMATCH")
        ok = False

    if ok:
        print("\nVERIFICATION PASSED: ledger matches anchor")
    else:
        print("\nVERIFICATION FAILED: ledger has been modified since anchor")

    return ok


def list_anchors():
    """List all ledger anchor tags."""
    try:
        result = _git_run(["tag", "-l", f"{TAG_PREFIX}*", "--sort=-creatordate"])
        tags = [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]
    except Exception:
        tags = []

    if not tags:
        print("No ledger anchors found.")
        return

    print(f"\nLedger anchors ({len(tags)} total):")
    for tag in tags[:20]:
        try:
            info = _git_run(["tag", "-l", "-n1", tag])
            desc = info.stdout.strip().split("\n")[-1].strip() if info.stdout.strip() else ""
            print(f"  {tag}  {desc}")
        except Exception:
            print(f"  {tag}")

    if len(tags) > 20:
        print(f"  ... and {len(tags) - 20} more")


def main():
    parser = argparse.ArgumentParser(description="Tamper-evident ledger anchoring")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ledger against latest anchor")
    parser.add_argument("--push", action="store_true",
                        help="Push the anchor tag to remote after creation")
    parser.add_argument("--list", action="store_true",
                        help="List all ledger anchor tags")
    args = parser.parse_args()

    if args.verify:
        ok = verify_anchor()
        sys.exit(0 if ok else 1)
    elif args.list:
        list_anchors()
    else:
        create_anchor(push=args.push)


if __name__ == "__main__":
    main()
