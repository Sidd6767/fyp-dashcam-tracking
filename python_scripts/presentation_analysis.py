"""Generate RAD presentation analyses: class performance, failures, curves, and speed."""
from pathlib import Path
import html
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "outputs"
NAMES = ["LMVs", "HMVs", "Road_Damages", "Unsurfaced_Roads", "Pedestrians", "Speed_Bumps"]
TEST_IMAGES = ROOT / "RAD" / "images" / "test" / "images"


def class_analysis():
    metrics = yaml.safe_load((OUT / "rad_model_comparison.yaml").read_text())
    values = metrics["YOLOv8s"]["per_class_map5095"]
    map50 = metrics["YOLOv8s"]["per_class_map50"]
    counts, sizes = [0] * 6, [[] for _ in NAMES]
    for label in (ROOT / "RAD" / "images" / "test" / "labels").glob("*.txt"):
        image = next((TEST_IMAGES / f"{label.stem}{ext}" for ext in (".jpg", ".jpeg", ".png") if (TEST_IMAGES / f"{label.stem}{ext}").exists()), None)
        if image is None:
            continue
        data = cv2.imread(str(image))
        if data is None:
            continue
        h, w = data.shape[:2]
        for row in label.read_text().splitlines():
            parts = row.split()
            if len(parts) == 5:
                cls, _, _, bw, bh = map(float, parts)
                if 0 <= int(cls) < 6:
                    counts[int(cls)] += 1
                    sizes[int(cls)].append(bw * w * bh * h)
    order = np.argsort(values)[::-1]
    lines = ["RAD PER-CLASS PERFORMANCE ANALYSIS", "", "Ranking (easy to hard by mAP@0.5:0.95):"]
    for rank, i in enumerate(order, 1):
        lines.append(f"{rank}. {NAMES[i]} | mAP50={map50[i]:.4f} | mAP50-95={values[i]:.4f} | instances={counts[i]} | box_area_px2={min(sizes[i], default=0):.0f}-{max(sizes[i], default=0):.0f}, median={np.median(sizes[i]) if sizes[i] else 0:.0f}")
    lines += ["", "HMVs are easiest because they are visually distinctive and generally larger.", "Road Damages are hardest because they are small, irregular, low contrast, and less distinctive."]
    (OUT / "rad_class_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([NAMES[i] for i in order], [values[i] for i in order], color="#4C78A8")
    ax.set_title("RAD Dataset Analysis - Oct 2026\nPer-Class Difficulty (easy to hard)")
    ax.set_ylabel("mAP@0.5:0.95"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "rad_class_performance_chart.png", dpi=180); plt.close(fig)


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(aa + ab - intersection, 1e-9)


def failure_cases():
    fail = OUT / "failure_cases"; fail.mkdir(exist_ok=True)
    model = YOLO(str(ROOT / "weights" / "rad_baseline_clear_best.pt"))
    candidates = []
    images = sorted(p for p in TEST_IMAGES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for index, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None: continue
        h, w = image.shape[:2]; ground_truth = []
        label = ROOT / "RAD" / "images" / "test" / "labels" / f"{path.stem}.txt"
        for row in label.read_text().splitlines() if label.exists() else []:
            p = row.split()
            if len(p) == 5:
                c, xc, yc, bw, bh = map(float, p)
                ground_truth.append((int(c), [(xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h]))
        result = model.predict(image, imgsz=640, device=0, conf=.15, verbose=False)[0]
        predictions = [(int(c), b, float(conf)) for b, conf, c in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())] if result.boxes is not None else []
        used = set(); matched = set()
        for gi, (gc, gb) in enumerate(ground_truth):
            choices = [(box_iou(gb, pb), pi) for pi, (pc, pb, _) in enumerate(predictions) if pc == gc and pi not in used]
            if choices:
                score, pi = max(choices)
                if score >= .5: matched.add(gi); used.add(pi)
        missed, false_positive = len(ground_truth)-len(matched), len(predictions)-len(used)
        lowest = min((p[2] for p in predictions), default=1.0)
        if missed or false_positive or lowest < .35:
            candidates.append((missed*3+false_positive*2+max(0, .5-lowest), path, image, ground_truth, predictions, missed, false_positive, lowest))
        if index % 100 == 0: print(f"Failure analysis: {index}/{len(images)}", flush=True)
    report = ["<html><head><meta charset='utf-8'><title>RAD Failure Cases</title></head><body><h1>RAD Detector Failure Cases</h1><p>Green: ground truth. Red: predictions.</p>"]
    for number, (_, path, image, truth, predictions, missed, fp, lowest) in enumerate(sorted(candidates, reverse=True)[:6], 1):
        for _, b in truth: cv2.rectangle(image, tuple(map(int,b[:2])), tuple(map(int,b[2:])), (0,180,0), 3)
        for cls, b, conf in predictions:
            cv2.rectangle(image, tuple(map(int,b[:2])), tuple(map(int,b[2:])), (0,0,220), 3)
            cv2.putText(image, f"{NAMES[cls]} {conf:.2f}", (int(b[0]), max(20,int(b[1])-5)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,220), 2)
        output = fail / f"failure_{number:02d}_{path.stem}.jpg"; cv2.imwrite(str(output), image)
        report.append(f"<section><h2>Case {number}: {html.escape(path.name)}</h2><p>{missed} false negatives, {fp} false positives, lowest confidence {lowest:.2f}</p><img src='failure_cases/{output.name}' width='900'></section>")
    (OUT / "failure_cases_report.html").write_text("\n".join(report+["</body></html>"]), encoding="utf-8")


def curves():
    data = pd.read_csv(ROOT / "results" / "training" / "rad_baseline_clear" / "results.csv"); epochs = data["epoch"] + 1
    fig, ax = plt.subplots(figsize=(9,5))
    for col, label, color in [("train/box_loss","Train box loss","#4C78A8"),("val/box_loss","Validation box loss","#F58518"),("train/cls_loss","Train class loss","#54A24B"),("val/cls_loss","Validation class loss","#E45756")]: ax.plot(epochs,data[col],label=label,color=color)
    ax.set_title("RAD Dataset Analysis - Oct 2026\nTraining and Validation Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT/"training_loss.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5)); ax.plot(epochs,data["metrics/mAP50(B)"],color="#4C78A8"); ax.set_title("RAD Dataset Analysis - Oct 2026\nValidation mAP@0.5"); ax.set_xlabel("Epoch"); ax.set_ylabel("mAP@0.5"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT/"training_map.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5)); ax.hist(data["metrics/precision(B)"],bins=8,color="#4C78A8",edgecolor="white"); ax.set_title("RAD Dataset Analysis - Oct 2026\nValidation Precision Distribution"); ax.set_xlabel("Precision"); ax.set_ylabel("Epoch count"); fig.tight_layout(); fig.savefig(OUT/"confidence_distribution.png",dpi=180); plt.close(fig)


def speed():
    weights = {"YOLOv8n": ROOT/"weights"/"rad_baseline_n_best.pt", "YOLOv8s": ROOT/"weights"/"rad_baseline_clear_best.pt", "YOLOv8m": ROOT/"weights"/"rad_baseline_m_best.pt"}
    images = sorted(TEST_IMAGES.glob("*.jpg"))[:20]; results = {}
    for name, weight in weights.items():
        model = YOLO(str(weight)); results[name] = {}
        for size in (320,640,1280):
            for image in images[:2]: model.predict(str(image),imgsz=size,device=0,verbose=False)
            times = []
            for image in images:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                start=time.perf_counter(); model.predict(str(image),imgsz=size,device=0,verbose=False)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
            results[name][size] = (float(np.mean(times)),1000/float(np.mean(times)))
    fig, ax = plt.subplots(figsize=(11,6)); labels=[f"{n}\n{s}px" for n,v in results.items() for s in v]; fps=[x[1] for v in results.values() for x in v.values()]
    ax.bar(labels,fps,color=["#4C78A8"]*3+["#F58518"]*3+["#54A24B"]*3); ax.axhline(30,color="#777",linestyle="--"); ax.set_title("RAD Dataset Analysis - Oct 2026\nInference FPS"); ax.set_ylabel("FPS"); fig.tight_layout(); fig.savefig(OUT/"speed_comparison.png",dpi=180); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    for name, function in [("class analysis",class_analysis),("failure cases",failure_cases),("training curves",curves),("speed benchmark",speed)]:
        print(f"Starting {name}", flush=True); function(); print(f"Completed {name}", flush=True)


if __name__ == "__main__":
    main()
