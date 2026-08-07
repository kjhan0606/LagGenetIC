#!/usr/bin/env python3
"""Summarize offline V-web and T-web diagnostics across parent snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def selected_target(path: Path, rank: int) -> dict[str, object]:
    document = json.loads(path.read_text())
    matches = [item for item in document["targets"] if int(item["mass_rank"]) == rank]
    if len(matches) != 1:
        raise ValueError(f"{path}: found {len(matches)} entries for rank {rank}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("time_series_root", type=Path)
    parser.add_argument("--target-rank", type=int, default=726)
    args = parser.parse_args()

    root = args.time_series_root.resolve()
    metric_paths = sorted(root.glob("output_*/classifier/web_classifier_metrics.json"))
    if not metric_paths:
        metric_paths = sorted(root.glob("[0-9]*/classifier/web_classifier_metrics.json"))
    rows: list[dict[str, object]] = []
    for metric_path in metric_paths:
        epoch_root = metric_path.parent.parent
        metric = json.loads(metric_path.read_text())
        watershed = selected_target(
            epoch_root / "watershed" / "watershed_properties.json", args.target_rank
        )
        rows.append(
            {
                "epoch": epoch_root.name,
                "aexp": metric["snapshot_info"]["aexp"],
                "minimum_delta": watershed["merged_minimum_delta"],
                "mean_delta": watershed["merged_mean_delta"],
                "effective_radius_mpc_h": watershed["merged_effective_radius_mpc_h"],
                "vweb_threshold": metric["vweb"]["threshold"],
                "vweb_precision": metric["vweb"]["precision"],
                "vweb_recall": metric["vweb"]["recall"],
                "vweb_f1": metric["vweb"]["f1"],
                "tweb_threshold": metric["tweb"]["threshold"],
                "tweb_precision": metric["tweb"]["precision"],
                "tweb_recall": metric["tweb"]["recall"],
                "tweb_f1": metric["tweb"]["f1"],
                "v_t_flag_jaccard": metric["flag_jaccard_with_each_other_in_scope"],
            }
        )
    rows.sort(key=lambda item: float(item["aexp"]))
    output = {
        "status": "diagnostic thresholds independently fitted at each epoch",
        "target_rank": args.target_rank,
        "rows": rows,
    }
    (root / "web_time_series_summary.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )

    aexp = np.asarray([item["aexp"] for item in rows], dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), constrained_layout=True)
    axes[0, 0].plot(
        aexp,
        [item["minimum_delta"] for item in rows],
        "o-",
        color="#4d4d4d",
        label=r"$\delta_{\min}$",
    )
    axes[0, 0].plot(
        aexp,
        [item["mean_delta"] for item in rows],
        "s--",
        color="#969696",
        label=r"$\langle\delta\rangle$",
    )
    axes[0, 0].set(ylabel="Watershed density contrast", xlabel="Scale factor")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        aexp,
        [item["effective_radius_mpc_h"] for item in rows],
        "o-",
        color="#984ea3",
    )
    axes[0, 1].set(
        ylabel=r"$R_{\rm eff}\ [h^{-1}\,{\rm Mpc}]$", xlabel="Scale factor"
    )

    axes[1, 0].plot(
        aexp,
        [item["vweb_threshold"] for item in rows],
        "o-",
        color="#377eb8",
        label="V-web",
    )
    axes[1, 0].plot(
        aexp,
        [item["tweb_threshold"] for item in rows],
        "s-",
        color="#e41a1c",
        label="T-web",
    )
    axes[1, 0].set(ylabel=r"Best-fit $\lambda_{\max}$ threshold", xlabel="Scale factor")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(
        aexp,
        [item["vweb_recall"] for item in rows],
        "o-",
        color="#377eb8",
        label="V recall",
    )
    axes[1, 1].plot(
        aexp,
        [item["tweb_recall"] for item in rows],
        "s-",
        color="#e41a1c",
        label="T recall",
    )
    axes[1, 1].plot(
        aexp,
        [item["v_t_flag_jaccard"] for item in rows],
        "^-",
        color="#4daf4a",
        label="V/T Jaccard",
    )
    axes[1, 1].set(ylabel="Fraction", xlabel="Scale factor", ylim=(0.0, 1.05))
    axes[1, 1].legend(frameon=False, ncol=2)

    for index, axis in enumerate(axes.ravel()):
        axis.grid(alpha=0.2)
        axis.text(
            0.02,
            0.97,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    figure.savefig(root / "web_time_series_summary.png", dpi=180)
    plt.close(figure)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
