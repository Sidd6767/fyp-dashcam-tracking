"""Evaluate RAD-trained model variants and create side-by-side comparison images."""
from pathlib import Path
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "outputs"
NAMES = ["LMVs", "HMVs", "Road_Damages", "Unsurfaced_Roads", "Pedestrians", "Speed_Bumps"]
MODELS = {"YOLOv8n": ROOT/"weights"/"rad_baseline_n_best.pt", "YOLOv8s": ROOT/"weights"/"rad_baseline_clear_best.pt", "YOLOv8m": ROOT/"weights"/"rad_baseline_m_best.pt"}


def evaluate():
    comparison = {}
    for name, weight in MODELS.items():
        model = YOLO(str(weight))
        result = model.val(data=str(ROOT/"RAD"/"data.yaml"), split="test", imgsz=640, batch=4 if name=="YOLOv8m" else 8, device=0, workers=0, plots=True, project=str(OUT/"evaluations"), name=f"{name.lower()}_clear_test_eval", exist_ok=True)
        box = result.box
        comparison[name] = {"map50":float(box.map50),"map5095":float(box.map),"precision":float(np.mean(box.p)),"recall":float(np.mean(box.r)),"per_class_map50":[float(x) for x in box.ap50],"per_class_map5095":[float(x) for x in box.maps]}
    (OUT/"rad_model_comparison.yaml").write_text(yaml.safe_dump(comparison,sort_keys=False),encoding="utf-8")


def comparison_images():
    target = OUT/"model_comparison_images"; target.mkdir(exist_ok=True)
    models = {name:YOLO(str(weight)) for name,weight in MODELS.items()}
    images = sorted((ROOT/"RAD"/"images"/"test"/"images").glob("*.jpg"))[:6]
    for path in images:
        image=cv2.imread(str(path)); panels=[]
        for name,model in models.items():
            panel=image.copy(); result=model.predict(image,imgsz=640,device=0,conf=.25,verbose=False)[0]
            if result.boxes is not None:
                for box,conf,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
                    x1,y1,x2,y2=map(int,box); cv2.rectangle(panel,(x1,y1),(x2,y2),(40,90,180),2); cv2.putText(panel,f"{NAMES[int(cls)]} {conf:.2f}",(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,90,180),1)
            cv2.putText(panel,name,(14,30),cv2.FONT_HERSHEY_SIMPLEX,.85,(30,30,30),2); panels.append(panel)
        height=720; panels=[cv2.resize(p,(int(p.shape[1]*height/p.shape[0]),height)) for p in panels]
        cv2.imwrite(str(target/f"{path.stem}_comparison.jpg"),np.hstack(panels),[cv2.IMWRITE_JPEG_QUALITY,90])


def main():
    OUT.mkdir(exist_ok=True); evaluate(); comparison_images(); print("Model evaluation and comparison images complete.")


if __name__ == "__main__":
    main()
