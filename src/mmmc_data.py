"""Utilities for loading, validating, and sampling the official MMMC dataset."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from datasets import Dataset, concatenate_datasets


REQUIRED_COLUMNS = {
    "image_id",
    "question",
    "answer",
    "conflict_type",
    "key_component",
    "key_component_relationships",
    "key_component_attributes",
    "image",
}


@dataclass(frozen=True)
class MMMCPair:
    """A matched conflict/clean pair that shares the same image."""

    image_id: int
    conflict_type: str
    conflict_index: int
    clean_index: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def load_arrow_split(split_dir: str | Path) -> Dataset:
    """Load all Arrow shards in a downloaded MMMC split."""

    split_dir = Path(split_dir)
    files = sorted(split_dir.glob("data-*.arrow"))
    if not files:
        raise FileNotFoundError(f"No Arrow shards found under {split_dir}")
    parts = [Dataset.from_file(str(path)) for path in files]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def build_and_validate_pairs(dataset: Dataset) -> list[MMMCPair]:
    """Validate the official pair structure and return one object per image.

    The MMMC test split is expected to contain exactly two rows per image:
    one row with a non-null conflict type and one matched clean row.
    """

    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Missing required MMMC columns: {sorted(missing)}")

    grouped: dict[int, list[tuple[int, str | None]]] = defaultdict(list)
    for index, (image_id, conflict_type) in enumerate(
        zip(dataset["image_id"], dataset["conflict_type"], strict=True)
    ):
        grouped[int(image_id)].append((index, conflict_type))

    pairs: list[MMMCPair] = []
    errors: list[str] = []
    for image_id, rows in grouped.items():
        conflict_rows = [(idx, kind) for idx, kind in rows if kind is not None]
        clean_rows = [(idx, kind) for idx, kind in rows if kind is None]
        if len(rows) != 2 or len(conflict_rows) != 1 or len(clean_rows) != 1:
            errors.append(
                f"image_id={image_id}: rows={len(rows)}, "
                f"conflict={len(conflict_rows)}, clean={len(clean_rows)}"
            )
            continue
        conflict_index, conflict_type = conflict_rows[0]
        clean_index, _ = clean_rows[0]
        pairs.append(
            MMMCPair(
                image_id=image_id,
                conflict_type=str(conflict_type),
                conflict_index=conflict_index,
                clean_index=clean_index,
            )
        )

    if errors:
        preview = "; ".join(errors[:10])
        raise ValueError(f"Invalid MMMC pair structure ({len(errors)} errors): {preview}")

    return sorted(pairs, key=lambda pair: pair.image_id)


def _proportional_quotas(group_sizes: dict[str, int], sample_size: int) -> dict[str, int]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    population = sum(group_sizes.values())
    if sample_size > population:
        raise ValueError(f"Requested {sample_size} pairs from a population of {population}")

    raw = {key: sample_size * size / population for key, size in group_sizes.items()}
    quotas = {key: math.floor(value) for key, value in raw.items()}
    remainder = sample_size - sum(quotas.values())

    order = sorted(
        group_sizes,
        key=lambda key: (raw[key] - quotas[key], group_sizes[key], key),
        reverse=True,
    )
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def stratified_sample(
    pairs: Sequence[MMMCPair],
    sample_size: int,
    seed: int,
) -> list[MMMCPair]:
    """Draw a deterministic proportional sample by conflict type."""

    groups: dict[str, list[MMMCPair]] = defaultdict(list)
    for pair in pairs:
        groups[pair.conflict_type].append(pair)
    quotas = _proportional_quotas(
        {kind: len(items) for kind, items in groups.items()},
        sample_size,
    )

    rng = random.Random(seed)
    selected: list[MMMCPair] = []
    for kind in sorted(groups):
        selected.extend(rng.sample(groups[kind], quotas[kind]))
    rng.shuffle(selected)
    return selected


def conflict_type_counts(pairs: Iterable[MMMCPair]) -> dict[str, int]:
    """Return stable conflict-type counts."""

    counts = Counter(pair.conflict_type for pair in pairs)
    return dict(sorted(counts.items()))

