"""Evaluate the trained detector candidates on the RAD test split."""

import csv
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "RAD" / "data.yaml"
OUTPUT_DIR = ROOT / "results" / "model_comparison"

MODELS = {
    "YOLOv8s": ROOT / "weights" / "rad_baseline_clear_best.pt",
    "YOLO11s": ROOT / "results" / "training" / "yolo11s_rad_25ep_gpu" / "weights" / "best.pt",
    "YOLO26s": ROOT / "results" / "training" / "yolo26s_rad_25ep_gpu" / "weights" / "best.pt",
}


def evaluate(name: str, weights: Path) -> dict:
    """Evaluate one detector and return standardized metrics."""
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(DATA),
        split="test",
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        plots=False,
        verbose=False,
    )

    start = time.perf_counter()
    for _ in model.predict(
        source=str(ROOT / "RAD" / "images" / "test" / "images"),
        imgsz=640,
        batch=1,
        device=0,
        workers=0,
        stream=True,
        verbose=False,
    ):
        pass
    elapsed = time.perf_counter() - start
    image_count = len(list((ROOT / "RAD" / "images" / "test" / "images").glob("*")))

    per_class = {
        class_name: float(ap)
        for class_name, ap in zip(metrics.names.values(), metrics.box.ap50)
    }
    result = {
        "model": name,
        "weights": str(weights.relative_to(ROOT)),
        "test_images": image_count,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_ap50": per_class,
        "inference_fps_including_batch_overhead": image_count / elapsed,
        "weight_size_mb": weights.stat().st_size / (1024 * 1024),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [evaluate(name, weights) for name, weights in MODELS.items()]

    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    fieldnames = [
        "model",
        "weights",
        "test_images",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "inference_fps_including_batch_overhead",
        "weight_size_mb",
    ]
    with (OUTPUT_DIR / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})

    lines = [
        "# Detector comparison",
        "",
        "| Model | Precision | Recall | mAP50 | mAP50-95 | FPS | Weight MB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {model} | {precision:.4f} | {recall:.4f} | {map50:.4f} | "
            "{map50_95:.4f} | {fps:.2f} | {size:.2f} |".format(
                model=result["model"],
                precision=result["precision"],
                recall=result["recall"],
                map50=result["map50"],
                map50_95=result["map50_95"],
                fps=result["inference_fps_including_batch_overhead"],
                size=result["weight_size_mb"],
            )
        )
    (OUTPUT_DIR / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for result in results:
        print(
            f"{result['model']}: mAP50={result['map50']:.4f}, "
            f"mAP50-95={result['map50_95']:.4f}, "
            f"FPS={result['inference_fps_including_batch_overhead']:.2f}"
        )


if __name__ == "__main__":
    main()
