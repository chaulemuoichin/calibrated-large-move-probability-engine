"""
One-command update: resolve pending forecasts and rebuild the site.

Usage:
    python scripts/update_live_verification.py
    python scripts/update_live_verification.py --publish    # also publish new forecasts
    python scripts/update_live_verification.py --demo       # demo mode
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Update live verification (resolve + rebuild site)")
    parser.add_argument("--publish", action="store_true",
                        help="Also publish new forecasts before resolving")
    parser.add_argument("--demo", action="store_true",
                        help="Use demo data (generate if needed)")
    parser.add_argument("--model-version", default="v1.0")
    args = parser.parse_args()

    if args.demo:
        # Generate demo data if not present
        demo_path = Path("outputs/live_verification/demo/forecasts.jsonl")
        if not demo_path.exists():
            print("Generating demo ledger...")
            from scripts.generate_demo_ledger import generate_demo
            generate_demo()

        print("\nBuilding site from demo data...")
        from scripts.build_live_verification_site import build_site
        build_site(demo=True)
        return

    # Live mode
    if args.publish:
        print("Publishing new forecasts...")
        from scripts.publish_live_forecasts import publish
        publish(model_version=args.model_version)

    print("\nResolving pending forecasts...")
    from scripts.resolve_live_forecasts import resolve
    resolve()

    print("\nBuilding site...")
    from scripts.build_live_verification_site import build_site
    build_site(demo=False)


if __name__ == "__main__":
    main()
