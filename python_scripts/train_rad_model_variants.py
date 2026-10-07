"""Train YOLOv8n and YOLOv8m sequentially on the corrected RAD dataset."""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "RAD" / "data.yaml"


def train(model_name: str, batch: int = 8) -> None:
    """Train one model and save its complete Ultralytics run separately."""
    print(f"\nStarting {model_name} training with batch={batch}", flush=True)
    model = YOLO(str(ROOT / "weights" / f"{model_name}.pt"))
    model.train(
        data=str(DATA),
        epochs=25,
        imgsz=640,
        batch=batch,
        device=0,
        workers=0,
        patience=5,
        save_period=5,
        project=str(ROOT / "results" / "training"),
        name=f"rad_baseline_{model_name[-1]}",
        exist_ok=True,
        verbose=True,
    )
    best = ROOT / "results" / "training" / f"rad_baseline_{model_name[-1]}" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"{model_name} training did not produce {best}")
    print(f"Completed {model_name}: {best}", flush=True)


def main() -> None:
    # These calls are intentionally sequential so GPU memory is released between runs.
    train("yolov8n", batch=8)
    train("yolov8m", batch=4)
    print("\nBoth RAD-trained model variants completed.", flush=True)


if __name__ == "__main__":
    main()
