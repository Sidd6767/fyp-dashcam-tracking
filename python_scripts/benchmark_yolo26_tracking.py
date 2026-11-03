"""Benchmark YOLO26s with ByteTrack on representative RAD videos.

The RAD videos do not include ground-truth track IDs, so MOTA, MOTP, and
ground-truth ID switches are reported as unavailable. This benchmark records
detector/tracker compatibility and prediction-only continuity proxies.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "results" / "weights" / "yolo26s_best.pt"
VIDEO_ROOT = ROOT / "RAD" / "videos_without_audio"
OUTPUT = ROOT / "results" / "tracking" / "yolo26_bytetrack_benchmark.json"
VIDEO_COUNT = 5
MAX_FRAMES = 300


def benchmark_video(model: YOLO, video_path: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()

    frame_count = 0
    detection_count = 0
    frames_with_tracks = 0
    track_frames: dict[int, list[int]] = defaultdict(list)
    start = time.perf_counter()
    try:
        results = model.track(
            source=str(video_path),
            tracker="bytetrack.yaml",
            persist=True,
            stream=True,
            conf=0.25,
            imgsz=640,
            device=0,
            verbose=False,
        )
        for result in results:
            if frame_count >= MAX_FRAMES:
                break
            boxes = result.boxes
            ids = boxes.id.int().cpu().tolist() if boxes is not None and boxes.id is not None else []
            detection_count += len(ids)
            if ids:
                frames_with_tracks += 1
            for track_id in ids:
                track_frames[int(track_id)].append(frame_count)
            frame_count += 1
    finally:
        elapsed = time.perf_counter() - start

    track_count = len(track_frames)
    continuity_values = []
    fragmented_tracks = 0
    for frames in track_frames.values():
        if len(frames) < 2:
            continuity_values.append(1.0)
            continue
        span = frames[-1] - frames[0] + 1
        continuity_values.append(len(frames) / span)
        if any(b - a > 1 for a, b in zip(frames, frames[1:])):
            fragmented_tracks += 1

    return {
        "video": str(video_path.relative_to(ROOT)),
        "source_fps": source_fps,
        "frames_processed": frame_count,
        "duration_seconds": frame_count / source_fps if source_fps else None,
        "detections_with_track_ids": detection_count,
        "unique_predicted_tracks": track_count,
        "frames_with_tracks_ratio": frames_with_tracks / frame_count if frame_count else 0.0,
        "track_continuity_ratio_mean": (
            sum(continuity_values) / len(continuity_values)
            if continuity_values
            else 0.0
        ),
        "fragmented_predicted_tracks": fragmented_tracks,
        "fragmentation_ratio": fragmented_tracks / track_count if track_count else 0.0,
        "processing_fps": frame_count / elapsed if elapsed else 0.0,
        "latency_ms_per_frame": elapsed * 1000 / frame_count if frame_count else None,
    }


def main() -> None:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"YOLO26 weights not found: {WEIGHTS}")
    videos = sorted(VIDEO_ROOT.rglob("*.mp4"))[:VIDEO_COUNT]
    if len(videos) < VIDEO_COUNT:
        raise FileNotFoundError(f"Expected {VIDEO_COUNT} videos under {VIDEO_ROOT}")

    model = YOLO(str(WEIGHTS))
    video_results = [benchmark_video(model, video) for video in videos]
    total_frames = sum(item["frames_processed"] for item in video_results)
    total_seconds = sum(item["duration_seconds"] or 0 for item in video_results)
    total_elapsed = sum(
        item["frames_processed"] / item["processing_fps"]
        for item in video_results
        if item["processing_fps"]
    )
    report = {
        "model": "YOLO26s",
        "tracker": "ByteTrack",
        "weights": str(WEIGHTS.relative_to(ROOT)),
        "videos_evaluated": len(video_results),
        "max_frames_per_video": MAX_FRAMES,
        "compatibility": "passed",
        "mota": None,
        "motp": None,
        "ground_truth_id_switches": None,
        "metrics_unavailable_reason": (
            "RAD videos do not include ground-truth track IDs or bounding-box "
            "annotations for these sequences."
        ),
        "aggregate": {
            "frames_processed": total_frames,
            "video_duration_seconds": total_seconds,
            "processing_fps": total_frames / total_elapsed if total_elapsed else 0.0,
            "latency_ms_per_frame": (
                total_elapsed * 1000 / total_frames if total_frames else None
            ),
            "mean_track_continuity_ratio": sum(
                item["track_continuity_ratio_mean"] for item in video_results
            ) / len(video_results),
            "total_fragmented_predicted_tracks": sum(
                item["fragmented_predicted_tracks"] for item in video_results
            ),
        },
        "videos": video_results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Saved report to {OUTPUT}")


if __name__ == "__main__":
    main()
