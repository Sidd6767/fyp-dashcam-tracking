# RAD Dashcam Object Detection System

## Project overview

This project is a vision-based object detection and tracking foundation for dashcam footage. Phase 1 detects six road-scene classes with YOLOv8 models trained on the RAD dataset. The trained detector can be used directly on videos with `detect_video.py`; the output is limited to the first 10 seconds and includes annotated video plus a CSV detection log.

The current project focus is clear-image object detection. Fog experiments and fog-specific outputs are not part of the active project.

## Dataset

Dataset location: `RAD/`

Dataset layout:

```text
RAD/
  data.yaml
  images/
    train/images/       5,843 images
    train/labels/
    valid/images/       1,288 images
    valid/labels/
    test/images/        1,263 images
    test/labels/
  videos_without_audio/
```

Total images: 8,394. Every image has a YOLO-format label. The six classes are:

1. LMVs
2. HMVs
3. Road_Damages
4. Unsurfaced_Roads
5. Pedestrians
6. Speed_Bumps

## Trained models

All three models were trained on the same corrected RAD labels for 25 epochs at 640 pixels.

| Model | Weights | Clear test mAP@0.5 | Clear test mAP@0.5:0.95 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| YOLOv8n | `weights/rad_baseline_n_best.pt` | 0.779847 | 0.439187 | 0.800612 | 0.709557 |
| **YOLOv8s** | `weights/rad_baseline_clear_best.pt` | **0.818978** | **0.467310** | **0.861887** | **0.749029** |
| YOLOv8m | `weights/rad_baseline_m_best.pt` | 0.792162 | 0.450984 | 0.839590 | 0.719239 |

YOLOv8s is the recommended model because it achieved the best overall clear-test accuracy and provides the best accuracy/size balance for this project.

## Per-class results

Clear-test mAP@0.5:0.95:

| Class | YOLOv8n | YOLOv8s | YOLOv8m |
|---|---:|---:|---:|
| LMVs | 0.600902 | 0.617979 | 0.615833 |
| HMVs | 0.707834 | 0.723659 | 0.727019 |
| Road_Damages | 0.204550 | 0.249912 | 0.236162 |
| Unsurfaced_Roads | 0.313311 | 0.361807 | 0.320669 |
| Pedestrians | 0.458607 | 0.501184 | 0.495705 |
| Speed_Bumps | 0.349917 | 0.349321 | 0.310513 |

HMVs are the easiest class because they are generally large and visually distinctive. Road Damages are the hardest because they can be small, irregular, low contrast, and visually similar to normal road texture.

## Inference speed

Measured on the configured NVIDIA RTX 3050 GPU using 20 clear RAD test images:

| Model | 320 px | 640 px | 1280 px |
|---|---:|---:|---:|
| YOLOv8n | 24.21 FPS | 21.00 FPS | 20.60 FPS |
| YOLOv8s | 25.20 FPS | 23.01 FPS | 17.35 FPS |
| YOLOv8m | 22.26 FPS | 21.25 FPS | 11.28 FPS |

The trained YOLOv8s model runs at approximately 23 FPS at 640 pixels on this machine. YOLOv8n is the lightweight alternative when lower model size is more important than accuracy.

## Running detection on a video

```powershell
python .\python_scripts\detect_video.py --video "D:\path\to\input.mp4"
```

The default model is `weights/rad_baseline_clear_best.pt`. The script processes at most 10 seconds and writes:

- An annotated MP4 video
- A CSV file containing frame number, timestamp, class, confidence, and bounding-box coordinates

Example with explicit output paths:

```powershell
python .\python_scripts\detect_video.py `
  --video "D:\input\road_video.mp4" `
  --output "D:\output\road_video_detected.mp4" `
  --summary "D:\output\road_video_detections.csv"
```

## Project folders

| Folder | Contents |
|---|---|
| `python_scripts/` | Active Python source files. |
| `weights/` | Pretrained weights and all RAD-trained checkpoints. |
| `results/` | Training logs, evaluation outputs, charts, videos, and presentation artifacts. |

Training logs are under `results/training/`; presentation charts and reports are under `results/outputs/`.

## Project scripts

| Script | Purpose |
|---|---|
| `python_scripts/detect_video.py` | Run the trained RAD detector on a video. |
| `python_scripts/train_rad_model_variants.py` | Train YOLOv8n and YOLOv8m sequentially. |
| `python_scripts/model_comparison.py` | Evaluate all three trained models and generate side-by-side predictions. |
| `python_scripts/presentation_analysis.py` | Generate class analysis, failure cases, training curves, and speed charts. |

## Presentation outputs

The `results/outputs/` folder contains:

- `rad_class_performance_chart.png`
- `training_loss.png`
- `training_map.png`
- `confidence_distribution.png`
- `speed_comparison.png`
- `model_metric_comparison.png`
- `failure_cases/`
- `failure_cases_report.html`
- `model_comparison_images/`

These files provide the charts, failure examples, and side-by-side model predictions needed for the midterm presentation.

## Recommended presentation conclusion

YOLOv8s is the best overall detector for this RAD experiment. It outperformed YOLOv8n and YOLOv8m on the clear test set, especially for Road Damages, Unsurfaced Roads, and Speed Bumps. The main remaining challenge is reliable detection of small or visually subtle road-damage objects. Future work can add object tracking, temporal association, and inference optimization for higher dashcam frame rates.
