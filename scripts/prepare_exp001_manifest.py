#!/usr/bin/env python3
"""Create nested, deterministic sample manifests for EXP-001."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mmmc_data import (  # noqa: E402
    build_and_validate_pairs,
    conflict_type_counts,
    load_arrow_split,
    stratified_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/exp001_baseline.yaml",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/mmmc/test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]
    base_seed = int(config["experiment"]["seed"])

    dataset = load_arrow_split(args.split_dir)
    all_pairs = build_and_validate_pairs(dataset)
    baseline = stratified_sample(
        all_pairs,
        int(dataset_config["baseline_pair_count"]),
        base_seed,
    )
    audit = stratified_sample(
        baseline,
        int(dataset_config["audit_pair_count"]),
        base_seed + 1,
    )
    smoke = stratified_sample(
        audit,
        int(dataset_config["smoke_pair_count"]),
        base_seed + 2,
    )

    audit_ids = {pair.image_id for pair in audit}
    smoke_ids = {pair.image_id for pair in smoke}
    manifest_path = PROJECT_ROOT / dataset_config["sample_manifest"]
    summary_path = PROJECT_ROOT / dataset_config["sample_summary"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for order, pair in enumerate(baseline):
            record = pair.to_dict()
            record.update(
                {
                    "baseline_order": order,
                    "in_audit": pair.image_id in audit_ids,
                    "in_smoke": pair.image_id in smoke_ids,
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "experiment_id": config["experiment"]["id"],
        "seed": base_seed,
        "dataset_revision": dataset_config["revision"],
        "dataset_rows": len(dataset),
        "population_pairs": len(all_pairs),
        "population_conflict_types": conflict_type_counts(all_pairs),
        "baseline_pairs": len(baseline),
        "baseline_rows": 2 * len(baseline),
        "baseline_conflict_types": conflict_type_counts(baseline),
        "audit_pairs": len(audit),
        "audit_rows": 2 * len(audit),
        "audit_conflict_types": conflict_type_counts(audit),
        "smoke_pairs": len(smoke),
        "smoke_rows": 2 * len(smoke),
        "smoke_conflict_types": conflict_type_counts(smoke),
        "nested": {
            "audit_subset_of_baseline": audit_ids.issubset(
                {pair.image_id for pair in baseline}
            ),
            "smoke_subset_of_audit": smoke_ids.issubset(audit_ids),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

