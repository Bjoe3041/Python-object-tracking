# UAV Quadcopter Detection

A real-time drone detection system built with YOLOv8, trained on a mix of real aerial photographs and procedurally generated synthetic composites. Started as a fun computer vision project, ended up working pretty well.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Results

| Metric | Score |
|--------|-------|
| Precision | **0.995** |
| Recall | **0.910** |
| F1 Score | **0.950** |

1639 true positives, 8 false positives, 163 false negatives across 1802 validation images.

## How it works

- **Detection** - YOLOv8s running inference on a live screen capture, drawing bounding boxes around detected drones in real time
- **Training data** - Real labeled drone photos from [Roboflow UAVs Hunting dataset](https://universe.roboflow.com/uavs-hunter/uavs-hunting/dataset/2), augmented with synthetically generated composites (drone cutouts pasted onto random backgrounds)
- **Synthetic generation** - Script that randomly scales and places drone objects onto background images, with HSV-based background removal to clean up object edges before compositing

## Project structure

```
Python object detection
├── ObjectDetection.py       # Live detection via screen capture
├── train_model.py           # Model training
├── generate_dataset.py      # Synthetic dataset generation + merges with real dataset
├── evaluate_model.py        # Evaluation + HTML report generation
├── clean_images.py          # HSV-based background removal for object cutouts
│
├── objects/                 # Drone cutout images used for synthetic generation
├── objects_cleaned/         # HSV-cleaned versions of the above
├── backgrounds/             # Background images for synthetic compositing
├── uav_dataset/             # Real labeled drone photos (Roboflow)
├── combination_dataset/     # Merged real + synthetic dataset (generated)
├── runs/                    # YOLO training output - weights, logs, charts
├── Models/                  # Manually saved model checkpoints
└── evaluation/              # Evaluation report and sample detection images
```

## Setup

```bash
git clone https://github.com/yourusername/uav-drone-detection
cd uav-drone-detection
python -m venv .venv
.venv\Scripts\activate
pip install ultralytics opencv-python numpy mss pillow matplotlib
```

You'll need to supply your own dataset - the real drone images used here are from the [Roboflow UAVs Hunting dataset](https://universe.roboflow.com/uavs-hunter/uavs-hunting/dataset/2) (MIT license). For synthetic generation, drop drone cutout images into `objects/` and background images into `backgrounds/`.

To run detection on your screen:
```bash
python ObjectDetection.py
```

To train from scratch:
```bash
python clean_images.py       # clean object cutout backgrounds
python generate_dataset.py   # generate synthetic data + merge with real dataset
python train_model.py        # train
python evaluate_model.py     # generate evaluation report
```

## Notes

- Trained on an RTX 2080, ~10 epochs, ~1-3 min/epoch, enough for experimenting
- The 163 false negatives are mostly small/distant drones - pushing `imgsz` to 1280 would likely improve recall further
- Synthetic data with noisy labels (unlabeled planes, helicopters) slightly deflates the real-world metrics - performance on clean data showed no false positives, but encountered more false negatives as well

## The next step for the project
Currently there are still a lot of false negatives, many of these can possibly be solved with better input data from the synthetic dataset.
Additionally, the synthetic generation creates a rectangular edge artifact where the object is pasted onto the background. The model picks this up as a feature, meaning any pasted image patch, that is perfectly rectangular, may get flagged as a drone regardless of content. Adding synthetic negatives - plain sky or noise patches pasted onto backgrounds with no label - should teach the model to ignore the artifact and focus on actual drone features, in theory.

## License

MIT - do whatever you want with it.