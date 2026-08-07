#!/usr/bin/env python3
"""Extract selected HOP mass-rank tiers for inverted-void cost comparisons."""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from select_parent_targets import Candidate, collect_candidate_ids


def parse_tier(specification: str) -> tuple[str, list[int]]:
    try:
        name, ranks_text = specification.split(":", maxsplit=1)
        ranks = [int(value) for value in ranks_text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "tier must have the form name:rank,rank,rank"
        ) from error
    if not name or not ranks or len(set(ranks)) != len(ranks) or min(ranks) <= 0:
        raise argparse.ArgumentTypeError("tier name and positive unique ranks are required")
    return name, ranks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ramses_output", type=Path)
    parser.add_argument("hop_tag", type=Path)
    parser.add_argument("candidate_catalogue", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tier", action="append", type=parse_tier, required=True)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--production-level", type=int, default=14)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir}: output directory is not empty")

    catalogue_path = args.candidate_catalogue.resolve()
    catalogue = json.loads(catalogue_path.read_text())
    by_rank = {entry["mass_rank"]: entry for entry in catalogue["candidates"]}
    requested: list[tuple[str, int, int]] = []
    seen_ranks: set[int] = set()
    for tier_name, ranks in args.tier:
        for tier_rank, mass_rank in enumerate(ranks, start=1):
            if mass_rank in seen_ranks:
                raise ValueError(f"mass rank {mass_rank} occurs in more than one tier")
            if mass_rank not in by_rank:
                raise ValueError(f"mass rank {mass_rank} is absent from the catalogue")
            seen_ranks.add(mass_rank)
            requested.append((tier_name, tier_rank, mass_rank))

    candidate_fields = {field.name for field in fields(Candidate)}
    candidates = [
        Candidate(**{key: value for key, value in by_rank[mass_rank].items() if key in candidate_fields})
        for _, _, mass_rank in requested
    ]
    memberships = collect_candidate_ids(
        args.ramses_output.resolve(),
        args.hop_tag.resolve(),
        candidates,
        args.grid_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_entries = []
    occupied: set[int] = set()
    level_factor = 8 ** (args.production_level - int(np.log2(args.grid_size)))
    for target_rank, ((tier_name, tier_rank, mass_rank), candidate, ids) in enumerate(
        zip(requested, candidates, memberships, strict=True), start=1
    ):
        overlap = occupied.intersection(map(int, ids))
        if overlap:
            raise ValueError(f"mass rank {mass_rank} overlaps another comparison target")
        occupied.update(map(int, ids))
        filename = (
            f"comparison_{target_rank:02d}_{tier_name}_rank_{mass_rank:06d}_"
            f"halo_{candidate.halo_id:06d}.id"
        )
        np.savetxt(output_dir / filename, ids, fmt="%d")
        entry = dict(by_rank[mass_rank])
        entry.update(
            {
                "selected": True,
                "target_rank": target_rank,
                "id_file": filename,
                "comparison_tier": tier_name,
                "tier_rank": tier_rank,
                "level14_selected_particles": int(candidate.particle_count * level_factor),
            }
        )
        selected_entries.append(entry)

    parameters = {
        "grid_size": args.grid_size,
        "production_level": args.production_level,
        "tiers": {name: ranks for name, ranks in args.tier},
        "source_catalogue": str(catalogue_path),
        "source_catalogue_sha256": sha256(catalogue_path),
        "hop_tag": str(args.hop_tag.resolve()),
        "hop_tag_sha256": sha256(args.hop_tag.resolve()),
        "prepared_at": datetime.now().astimezone().isoformat(),
    }
    (output_dir / "parent_target_candidates.json").write_text(
        json.dumps({"parameters": parameters, "candidates": selected_entries}, indent=2)
        + "\n"
    )
    with (output_dir / "candidate_comparison.tsv").open("w") as handle:
        handle.write(
            "target_rank\ttier\ttier_rank\tmass_rank\thalo_id\tnpart\t"
            "max_lagrangian_width_mpc_h\tlevel14_selected_particles\n"
        )
        for entry in selected_entries:
            handle.write(
                f"{entry['target_rank']}\t{entry['comparison_tier']}\t"
                f"{entry['tier_rank']}\t{entry['mass_rank']}\t{entry['halo_id']}\t"
                f"{entry['particle_count']}\t"
                f"{max(entry['lagrangian_width']) * args.grid_size:.6f}\t"
                f"{entry['level14_selected_particles']}\n"
            )
    for entry in selected_entries:
        print(
            f"{entry['comparison_tier']} rank {entry['mass_rank']}: "
            f"HOP {entry['halo_id']}, N={entry['particle_count']}, "
            f"level-14 particles={entry['level14_selected_particles']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
