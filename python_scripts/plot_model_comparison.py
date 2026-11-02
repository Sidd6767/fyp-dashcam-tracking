"""Create graphs comparing the trained detector models."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
COMPARISON = ROOT / "results" / "model_comparison" / "comparison.csv"
OUTPUT = ROOT / "results" / "model_comparison" / "model_comparison_graph.png"
PARAMETERS = {
    "YOLOv8s": 11_127_906,
    "YOLO11s": 9_430_114,
    "YOLO26s": 9_952_508,
}


def main() -> None:
    with COMPARISON.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    models = [row["model"] for row in rows]
    metrics = {
        "mAP@0.5": [float(row["map50"]) for row in rows],
        "FPS": [float(row["inference_fps_including_batch_overhead"]) for row in rows],
        "Parameters (millions)": [
            PARAMETERS[model] / 1_000_000 for model in models
        ],
    }

    colors = ["#2563eb", "#f59e0b", "#16a34a"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 5))
    figure.suptitle("RAD Test-Set Detector Comparison", fontsize=16, fontweight="bold")

    for axis, (title, values) in zip(axes, metrics.items()):
        bars = axis.bar(models, values, color=colors)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", rotation=20)
        if title == "mAP@0.5":
            axis.set_ylim(0, 1)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}" if title == "mAP@0.5" else (
                    f"{value:.1f}" if title == "FPS" else f"{value:.2f}"
                ),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    figure.text(
        0.5,
        0.02,
        "Evaluation uses 1,263 RAD test images at 640 px on an RTX 3050",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved graph to {OUTPUT}")


if __name__ == "__main__":
    main()
