"""Run RAD object detection on a video, limited to the first 10 seconds.

Example:
    python detect_video.py --video "path\\to\\input.mp4"
    python detect_video.py --video "path\\to\\input.mp4" --output detected.mp4 --conf 0.30
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_NAMES = [
    "LMVs",
    "HMVs",
    "Road_Damages",
    "Unsurfaced_Roads",
    "Pedestrians",
    "Speed_Bumps",
]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "rad_baseline_clear_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAD YOLO object detection on the first 10 seconds of a video."
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"YOLO weights path (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated output video path (default: <input>_detected.mp4).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="CSV detection summary path (default: <output stem>_detections.csv).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--device",
        default="0",
        help="Inference device, for example 0 for the first GPU or cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.resolve()
    weights_path = args.weights.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {weights_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1.")

    output_path = (
        args.output.resolve()
        if args.output
        else video_path.with_name(f"{video_path.stem}_detected.mp4")
    )
    summary_path = (
        args.summary.resolve()
        if args.summary
        else output_path.with_name(f"{output_path.stem}_detections.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_limit = max(1, int(round(fps * 10.0)))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(frame_limit, total_frames) if total_frames > 0 else frame_limit

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    model = YOLO(str(weights_path))
    rows: list[dict[str, object]] = []
    frame_number = 0
    try:
        while frame_number < frames_to_process:
            success, frame = capture.read()
            if not success:
                break
            result = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            annotated = result.plot(
                labels=True,
                conf=True,
                boxes=True,
            )
            writer.write(annotated)

            if result.boxes is not None:
                boxes = result.boxes
                coordinates = boxes.xyxy.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                classes = boxes.cls.cpu().tolist()
                for box, confidence, class_id in zip(coordinates, confidences, classes):
                    class_index = int(class_id)
                    rows.append(
                        {
                            "frame": frame_number,
                            "time_seconds": round(frame_number / fps, 4),
                            "class_id": class_index,
                            "class_name": CLASS_NAMES[class_index],
                            "confidence": round(float(confidence), 6),
                            "x_min": round(float(box[0]), 2),
                            "y_min": round(float(box[1]), 2),
                            "x_max": round(float(box[2]), 2),
                            "y_max": round(float(box[3]), 2),
                        }
                    )
            frame_number += 1
    finally:
        capture.release()
        writer.release()

    fieldnames = [
        "frame",
        "time_seconds",
        "class_id",
        "class_name",
        "confidence",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        csv_writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    duration = frame_number / fps
    print(f"Input: {video_path}")
    print(f"Model: {weights_path}")
    print(f"Frames processed: {frame_number}")
    print(f"Output duration: {duration:.3f} seconds (maximum 10 seconds)")
    print(f"Detections: {len(rows)}")
    print(f"Annotated video: {output_path}")
    print(f"Detection CSV: {summary_path}")


if __name__ == "__main__":
    main()
