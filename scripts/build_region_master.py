#!/usr/bin/env python3
"""
Offline preprocessing script: build the canonical region master artifact
from a local extraction of the cahyadsn/wilayah ZIP, ONCE, so the
dashboard never has to re-download or re-parse the raw SQL dump on every
page load (project-owner decision).

Usage:
    python scripts/build_region_master.py \
        --wilayah-sql path/to/db/wilayah.sql \
        --level12-sql path/to/db/wilayah_level_1_2.sql \
        --out region_master_canonical.csv

Note on GOOGLE_DRIVE_REGION_FOLDER_ID: this script currently takes local
file paths, not a live Drive fetch. app/data/drive_adapter.py does not yet
have a "region" source key (docs/11 only defines DATA_DRIVE/POLICY_DRIVE);
wiring GOOGLE_DRIVE_REGION_FOLDER_ID to an actual download here is a known
remaining gap -- once added, this script's file paths would come from that
download step instead of CLI arguments, and this file's core logic
(build + save) would stay the same.
"""
import argparse
import sys

sys.path.insert(0, ".")

from app.data.region_master import build_canonical_region_master, save_canonical_artifact
from config.settings import load_region_source_config, ConfigError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wilayah-sql", required=True, help="path to db/wilayah.sql")
    parser.add_argument("--level12-sql", required=True, help="path to db/wilayah_level_1_2.sql")
    parser.add_argument("--out", default="region_master_canonical.csv")
    args = parser.parse_args()

    try:
        region_config = load_region_source_config(required=False)
        if region_config.region_folder_id:
            print(f"[info] GOOGLE_DRIVE_REGION_FOLDER_ID is set ({region_config.region_folder_id}), "
                  "but live Drive fetch is not wired yet -- using the local file paths given.")
    except ConfigError:
        pass

    print(f"[info] parsing {args.wilayah_sql} and {args.level12_sql} ...")
    region_master = build_canonical_region_master(args.wilayah_sql, args.level12_sql)
    save_canonical_artifact(region_master, args.out)

    n_geo = int(region_master["has_geometry"].sum())
    print(f"[ok] wrote {args.out}: {len(region_master)} kabupaten/kota rows, "
          f"{n_geo} with geometry ({n_geo/len(region_master):.1%}).")


if __name__ == "__main__":
    main()
