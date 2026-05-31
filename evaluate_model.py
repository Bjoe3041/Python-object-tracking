"""
evaluate_model.py
Runs inference on the validation set, computes stats, saves charts,
and writes a summary
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from ultralytics import YOLO
import base64


# ---------------- CONFIG ----------------
MODEL_PATH   = "runs/detect/train6/weights/best.pt"
DATA_YAML    = "combination_dataset/data.yaml"
VAL_IMAGES   = "combination_dataset/images/val"
VAL_LABELS   = "combination_dataset/labels/val"
OUT_DIR      = "evaluation"
CONF_THRESH  = 0.25   # confidence threshold for predictions
IOU_THRESH   = 0.3    # IoU threshold to count a detection as correct
# ----------------------------------------

CHARTS_DIR = os.path.join(OUT_DIR, "charts")
SAMPLES_DIR = os.path.join(OUT_DIR, "samples")

def setup():
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter

    standard_iou = inter / union if union > 0 else 0

    # How much of A is inside B, and how much of B is inside A
    containment_a = inter / areaA if areaA > 0 else 0  # pred contained in GT
    containment_b = inter / areaB if areaB > 0 else 0  # GT contained in pred

    return max(standard_iou, containment_a, containment_b)


def xywhn_to_xyxy(box, w, h):
    """Convert YOLO normalised xywh to pixel x1y1x2y2."""
    cx, cy, bw, bh = box
    x1 = (cx - bw/2) * w
    y1 = (cy - bh/2) * h
    x2 = (cx + bw/2) * w
    y2 = (cy + bh/2) * h
    return [x1, y1, x2, y2]


def load_ground_truth(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append(xywhn_to_xyxy(list(map(float, parts[1:])), img_w, img_h))
    return boxes


def run_evaluation(model):
    img_files = [f for f in os.listdir(VAL_IMAGES)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    TPs, FPs, FNs = 0, 0, 0
    confidences_tp = []
    confidences_fp = []
    all_confs = []
    per_image_recall = []
    size_buckets = {"small (<5%)": [0,0], "medium (5-15%)": [0,0], "large (>15%)": [0,0]}

    sample_tp, sample_fp, sample_fn = [], [], []

    print(f"Evaluating {len(img_files)} images...")

    for idx, fname in enumerate(img_files):
        img_path = os.path.join(VAL_IMAGES, fname)
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(VAL_LABELS, stem + ".txt")

        results = model(img_path, conf=CONF_THRESH, verbose=False, iou=0.3)
        result = results[0]

        img_h, img_w = result.orig_shape
        gt_boxes = load_ground_truth(label_path, img_w, img_h)

        pred_boxes = []
        pred_confs = []
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                pred_boxes.append(box.xyxy[0].tolist())
                pred_confs.append(float(box.conf[0]))
                all_confs.append(float(box.conf[0]))

        def nms_manual(boxes, confs, iou_thresh=0.3):
            if not boxes:
                return boxes, confs
            order = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
            keep = []
            while order:
                i = order.pop(0)
                keep.append(i)
                order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thresh]
            return [boxes[i] for i in keep], [confs[i] for i in keep]

        pred_boxes, pred_confs = nms_manual(pred_boxes, pred_confs, iou_thresh=0.3)

        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()

        for pi, pb in enumerate(pred_boxes):
            best_iou = 0
            best_gi = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                score = iou(pb, gb)
                if score > best_iou:
                    best_iou = score
                    best_gi = gi
            if best_iou >= IOU_THRESH and best_gi != -1:
                TPs += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                confidences_tp.append(pred_confs[pi])
                # Size bucket
                gb = gt_boxes[best_gi]
                area_pct = ((gb[2]-gb[0]) * (gb[3]-gb[1])) / (img_w * img_h) * 100
                if area_pct < 5:
                    size_buckets["small (<5%)"][0] += 1
                elif area_pct < 15:
                    size_buckets["medium (5-15%)"][0] += 1
                else:
                    size_buckets["large (>15%)"][0] += 1
            else:
                FPs += 1
                confidences_fp.append(pred_confs[pi])

        img_fn = len(gt_boxes) - len(matched_gt)
        FNs += img_fn

        # Per-image recall
        if gt_boxes:
            recall = len(matched_gt) / len(gt_boxes)
            per_image_recall.append(recall)
            # Size buckets total
            for gi, gb in enumerate(gt_boxes):
                area_pct = ((gb[2]-gb[0]) * (gb[3]-gb[1])) / (img_w * img_h) * 100
                if area_pct < 5:
                    size_buckets["small (<5%)"][1] += 1
                elif area_pct < 15:
                    size_buckets["medium (5-15%)"][1] += 1
                else:
                    size_buckets["large (>15%)"][1] += 1

        # Collect sample images
        has_tp = len(matched_gt) > 0
        has_fp = len(pred_boxes) - len(matched_pred) > 0
        has_fn = img_fn > 0

        if has_tp and len(sample_tp) < 3:
            sample_tp.append((img_path, gt_boxes, pred_boxes, pred_confs, matched_pred))
        if has_fp and len(sample_fp) < 3:
            sample_fp.append((img_path, gt_boxes, pred_boxes, pred_confs, matched_pred))
        if has_fn and len(sample_fn) < 3:
            sample_fn.append((img_path, gt_boxes, pred_boxes, pred_confs, matched_pred))

        if idx % 100 == 0:
            print(f"  {idx}/{len(img_files)}")

    precision = TPs / (TPs + FPs) if (TPs + FPs) > 0 else 0
    recall    = TPs / (TPs + FNs) if (TPs + FNs) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    stats = {
        "total_images": len(img_files),
        "TP": TPs, "FP": FPs, "FN": FNs,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_conf_tp": float(np.mean(confidences_tp)) if confidences_tp else 0,
        "avg_conf_fp": float(np.mean(confidences_fp)) if confidences_fp else 0,
        "avg_image_recall": float(np.mean(per_image_recall)) if per_image_recall else 0,
        "confidences_tp": confidences_tp,
        "confidences_fp": confidences_fp,
        "per_image_recall": per_image_recall,
        "size_buckets": size_buckets,
        "samples": {
            "tp": sample_tp,
            "fp": sample_fp,
            "fn": sample_fn,
        }
    }
    return stats


# ---- CHARTS ----

def chart_confusion(stats):
    fig, ax = plt.subplots(figsize=(5, 4))
    tp, fp, fn = stats["TP"], stats["FP"], stats["FN"]
    tn = 0  # not tracked in single-class detection
    matrix = np.array([[tp, fn], [fp, tn]])
    labels = [["TP", "FN"], ["FP", "TN*"]]
    colors = [["#4CAF50", "#F44336"], ["#FF9800", "#9E9E9E"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, color=colors[i][j], alpha=0.8))
            ax.text(j+0.5, 1.5-i, f"{labels[i][j]}\n{matrix[i][j]}",
                    ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["Predicted\nDrone", "Predicted\nBackground"])
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["Actual\nBackground", "Actual\nDrone"])
    ax.set_title("Confusion Matrix\n(*TN not tracked in detection)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def chart_prf(stats):
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics = ["Precision", "Recall", "F1"]
    values  = [stats["precision"], stats["recall"], stats["f1"]]
    colors  = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(metrics, values, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_title("Precision / Recall / F1")
    ax.set_ylabel("Score")
    ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5, label='0.8 target')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "precision_recall_f1.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def chart_conf_dist(stats):
    fig, ax = plt.subplots(figsize=(6, 4))
    if stats["confidences_tp"]:
        ax.hist(stats["confidences_tp"], bins=20, alpha=0.7, color="#4CAF50", label="True Positives")
    if stats["confidences_fp"]:
        ax.hist(stats["confidences_fp"], bins=20, alpha=0.7, color="#F44336", label="False Positives")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Score Distribution")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "confidence_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def chart_size_recall(stats):
    buckets = stats["size_buckets"]
    labels, recalls = [], []
    for name, (correct, total) in buckets.items():
        labels.append(name)
        recalls.append(correct / total if total > 0 else 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#F44336", "#FF9800", "#4CAF50"]
    bars = ax.bar(labels, recalls, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_title("Recall by Object Size")
    ax.set_ylabel("Recall")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "recall_by_size.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def chart_image_recall_hist(stats):
    recalls = stats["per_image_recall"]
    if not recalls:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(recalls, bins=10, range=(0, 1), color="#2196F3", edgecolor='white', linewidth=1.2)
    ax.set_xlabel("Per-image Recall")
    ax.set_ylabel("Number of Images")
    ax.set_title("Per-image Recall Distribution")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "per_image_recall.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def save_sample_images(stats):
    """Save annotated sample images for TP, FP, FN cases."""
    from PIL import Image, ImageDraw, ImageFont

    def draw_boxes(img_path, gt_boxes, pred_boxes, pred_confs, matched_pred, title):
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        # Ground truth = green
        for gb in gt_boxes:
            draw.rectangle(gb, outline=(0, 220, 0), width=3)
        # Predictions
        for pi, (pb, conf) in enumerate(zip(pred_boxes, pred_confs)):
            color = (0, 150, 255) if pi in matched_pred else (255, 50, 50)
            draw.rectangle(pb, outline=color, width=3)
            draw.text((pb[0]+2, pb[1]+2), f"{conf:.2f}", fill=color)
        return img

    saved = {}
    for kind, samples in stats["samples"].items():
        paths = []
        for i, sample in enumerate(samples):
            img_path, gt_boxes, pred_boxes, pred_confs, matched_pred = sample
            img = draw_boxes(img_path, gt_boxes, pred_boxes, pred_confs, matched_pred, kind)
            out = os.path.join(SAMPLES_DIR, f"{kind}_{i+1}.jpg")
            img.save(out, quality=90)
            paths.append(out)
        saved[kind] = paths
    return saved


# Project summary

def img_to_base64(path, mime="image/png"):
    """Embed an image file as a base64 data URI."""
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def write_report(stats, chart_paths, sample_paths):
    p  = stats["precision"]
    r  = stats["recall"]
    f1 = stats["f1"]

    def rating(val):
        if val >= 0.85: return ("Excellent", "#00e5a0")
        if val >= 0.70: return ("Decent",    "#f5c542")
        return ("Needs work", "#ff5c5c")

    def metric_card(label, value, suffix=""):
        tag, color = rating(value)
        return f"""
        <div class="card">
          <div class="card-label">{label}</div>
          <div class="card-value" style="color:{color}">{value:.3f}{suffix}</div>
          <div class="card-tag" style="background:{color}22; color:{color}">{tag}</div>
        </div>"""

    def count_card(label, value, color, desc):
        return f"""
        <div class="count-card">
          <div class="count-num" style="color:{color}">{value}</div>
          <div class="count-label">{label}</div>
          <div class="count-desc">{desc}</div>
        </div>"""

    def chart_section(title, desc, path, mime="image/png"):
        src = img_to_base64(path, mime)
        if not src:
            return ""
        return f"""
        <div class="chart-block">
          <div class="chart-meta">
            <h3>{title}</h3>
            <p>{desc}</p>
          </div>
          <img src="{src}" alt="{title}" class="chart-img" />
        </div>"""

    def sample_row(kind, label, color, paths):
        imgs = ""
        for p in paths:
            src = img_to_base64(p, "image/jpeg")
            if src:
                imgs += f'<img src="{src}" alt="{label}" class="sample-img" />'
        if not imgs:
            return ""
        return f"""
        <div class="sample-group">
          <div class="sample-tag" style="background:{color}22; color:{color}; border:1px solid {color}44">{label}</div>
          <div class="sample-row">{imgs}</div>
        </div>"""

    size_rows = ""
    for name, (correct, total) in stats["size_buckets"].items():
        recall_val = correct / total if total > 0 else 0
        bar_color = "#00e5a0" if recall_val >= 0.8 else "#f5c542" if recall_val >= 0.6 else "#ff5c5c"
        size_rows += f"""
        <div class="size-row">
          <span class="size-label">{name}</span>
          <div class="size-bar-wrap">
            <div class="size-bar" style="width:{recall_val*100:.0f}%; background:{bar_color}"></div>
          </div>
          <span class="size-pct">{recall_val:.2f} ({correct}/{total})</span>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Drone Detection — Project Report</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:       #0a0c10;
    --surface:  #111318;
    --border:   #1e2128;
    --text:     #c8cdd8;
    --muted:    #555d6e;
    --accent:   #00e5a0;
    --accent2:  #0098ff;
    --danger:   #ff5c5c;
    --warn:     #f5c542;
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    line-height: 1.7;
    min-height: 100vh;
  }}

  /* ── HERO ── */
  .hero {{
    position: relative;
    padding: 80px 60px 60px;
    border-bottom: 1px solid var(--border);
    overflow: hidden;
  }}
  .hero::before {{
    content: '';
    position: absolute;
    inset: 0;
    background:
      radial-gradient(ellipse 60% 80% at 80% 50%, #00e5a015 0%, transparent 70%),
      radial-gradient(ellipse 40% 60% at 10% 80%, #0098ff0a 0%, transparent 60%);
    pointer-events: none;
  }}
  .hero-eyebrow {{
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 16px;
  }}
  .hero h1 {{
    font-family: 'Syne', sans-serif;
    font-size: clamp(32px, 5vw, 64px);
    font-weight: 800;
    line-height: 1.05;
    color: #fff;
    max-width: 700px;
  }}
  .hero h1 span {{ color: var(--accent); }}
  .hero-sub {{
    margin-top: 20px;
    color: var(--muted);
    max-width: 560px;
    font-size: 13px;
  }}
  .hero-meta {{
    margin-top: 36px;
    display: flex;
    gap: 32px;
    flex-wrap: wrap;
  }}
  .meta-item {{
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .meta-key {{
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
  }}
  .meta-val {{
    font-size: 13px;
    color: var(--text);
  }}
  .meta-val code {{
    background: var(--border);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    color: var(--accent2);
  }}

  /* ── LAYOUT ── */
  .container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 60px 40px;
  }}
  section {{ margin-bottom: 80px; }}
  .section-title {{
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }}

  /* ── METRIC CARDS ── */
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    transition: border-color 0.2s;
  }}
  .card:hover {{ border-color: #2e3340; }}
  .card-label {{
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
  }}
  .card-value {{
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
  }}
  .card-tag {{
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 999px;
    display: inline-block;
    width: fit-content;
    letter-spacing: 0.05em;
  }}

  /* ── COUNT CARDS ── */
  .counts {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
  }}
  .count-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 24px;
    text-align: center;
  }}
  .count-num {{
    font-family: 'Syne', sans-serif;
    font-size: 48px;
    font-weight: 800;
    line-height: 1;
  }}
  .count-label {{
    margin-top: 8px;
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text);
  }}
  .count-desc {{
    margin-top: 6px;
    font-size: 11px;
    color: var(--muted);
  }}

  /* ── CHARTS ── */
  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }}
  .chart-block {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }}
  .chart-block.full {{ grid-column: 1 / -1; }}
  .chart-meta {{
    padding: 20px 24px 0;
  }}
  .chart-meta h3 {{
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 4px;
  }}
  .chart-meta p {{
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 16px;
  }}
  .chart-img {{
    width: 100%;
    display: block;
  }}

  /* ── SIZE BARS ── */
  .size-table {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }}
  .size-row {{
    display: grid;
    grid-template-columns: 160px 1fr 120px;
    align-items: center;
    gap: 16px;
  }}
  .size-label {{ font-size: 12px; color: var(--text); }}
  .size-bar-wrap {{
    height: 8px;
    background: var(--border);
    border-radius: 999px;
    overflow: hidden;
  }}
  .size-bar {{
    height: 100%;
    border-radius: 999px;
    transition: width 1s ease;
  }}
  .size-pct {{
    font-size: 12px;
    color: var(--muted);
    text-align: right;
  }}

  /* ── SAMPLES ── */
  .sample-group {{ margin-bottom: 32px; }}
  .sample-tag {{
    display: inline-block;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 999px;
    margin-bottom: 16px;
  }}
  .sample-row {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .sample-img {{
    height: 200px;
    width: auto;
    border-radius: 8px;
    border: 1px solid var(--border);
    object-fit: cover;
    flex: 1;
    min-width: 180px;
    max-width: 380px;
  }}

  /* ── INTERPRETATION TABLE ── */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  th {{
    text-align: left;
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: #ffffff04; }}
  td:first-child {{ color: #fff; font-weight: 500; }}
  td code {{
    background: var(--border);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 12px;
    color: var(--accent2);
  }}

  /* ── FOOTER ── */
  .footer {{
    border-top: 1px solid var(--border);
    padding: 32px 60px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 0.08em;
  }}
  .footer span {{ color: var(--accent); }}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-eyebrow">Computer Vision · Object Detection · YOLOv8</div>
  <h1>UAV <span>Drone</span><br>Detection System</h1>
  <p class="hero-sub">
    A real-time drone detection model trained on a combined dataset of real aerial photographs
    and procedurally generated synthetic composites, evaluated against {stats['total_images']} validation images.
  </p>
  <div class="hero-meta">
    <div class="meta-item">
      <span class="meta-key">Model</span>
      <span class="meta-val"><code>{MODEL_PATH}</code></span>
    </div>
    <div class="meta-item">
      <span class="meta-key">Conf threshold</span>
      <span class="meta-val">{CONF_THRESH}</span>
    </div>
    <div class="meta-item">
      <span class="meta-key">IoU threshold</span>
      <span class="meta-val">{IOU_THRESH}</span>
    </div>
    <div class="meta-item">
      <span class="meta-key">Val images</span>
      <span class="meta-val">{stats['total_images']}</span>
    </div>
  </div>
</div>

<div class="container">

  <section>
    <div class="section-title">Performance Metrics</div>
    <div class="cards">
      {metric_card("Precision", stats["precision"])}
      {metric_card("Recall", stats["recall"])}
      {metric_card("F1 Score", stats["f1"])}
      <div class="card">
        <div class="card-label">Avg Conf · TP</div>
        <div class="card-value" style="color:#00e5a0">{stats['avg_conf_tp']:.3f}</div>
        <div class="card-tag" style="background:#00e5a022;color:#00e5a0">True Positives</div>
      </div>
      <div class="card">
        <div class="card-label">Avg Conf · FP</div>
        <div class="card-value" style="color:#ff5c5c">{stats['avg_conf_fp']:.3f}</div>
        <div class="card-tag" style="background:#ff5c5c22;color:#ff5c5c">False Positives</div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-title">Detection Counts</div>
    <div class="counts">
      {count_card("True Positives", stats['TP'], "#00e5a0", "Drones correctly detected")}
      {count_card("False Positives", stats['FP'], "#f5c542", "Background mistaken for drone")}
      {count_card("False Negatives", stats['FN'], "#ff5c5c", "Drones missed entirely")}
    </div>
  </section>

  <section>
    <div class="section-title">Charts &amp; Analysis</div>
    <div class="charts-grid">
      {chart_section("Confusion Matrix", "Breakdown of prediction outcomes across all validation images.", chart_paths.get("confusion"))}
      {chart_section("Precision / Recall / F1", "Core detection quality metrics. Precision measures accuracy of positive predictions; recall measures coverage.", chart_paths.get("prf"))}
      {chart_section("Confidence Distribution", "Ideally TP confidences cluster high (>0.7) while FP confidences stay low — showing the model knows when it's certain.", chart_paths.get("conf_dist"))}
      {chart_section("Per-image Recall", "How completely the model detects drones within each image. Images near 1.0 had all drones found.", chart_paths.get("img_recall"))}
    </div>
  </section>

  <section>
    <div class="section-title">Detection by Object Size</div>
    <div class="size-table">
      {size_rows}
    </div>
  </section>

  <section>
    <div class="section-title">Sample Detections</div>
    <p style="color:var(--muted); font-size:12px; margin-bottom:28px;">
      Green boxes = ground truth labels &nbsp;·&nbsp;
      Blue boxes = correct predictions &nbsp;·&nbsp;
      Red boxes = false positives &nbsp;·&nbsp;
      Numbers = confidence score
    </p>
    {sample_row("tp", "✓ True Positives — correct detections", "#00e5a0", sample_paths.get("tp", []))}
    {sample_row("fp", "✗ False Positives — false alarms", "#f5c542", sample_paths.get("fp", []))}
    {sample_row("fn", "○ False Negatives — missed drones", "#ff5c5c", sample_paths.get("fn", []))}
  </section>

  <section>
    <div class="section-title">Interpretation Guide</div>
    <div style="background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden;">
      <table>
        <thead>
          <tr>
            <th>Problem</th>
            <th>What it means</th>
            <th>How to fix</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Low recall</td>
            <td>Missing drones in the scene</td>
            <td>More training data, bigger model, higher <code>imgsz</code></td>
          </tr>
          <tr>
            <td>Low precision</td>
            <td>Too many false alarms</td>
            <td>Harder negative examples, more background variety in training</td>
          </tr>
          <tr>
            <td>Bad on small objects</td>
            <td>Model struggles at range</td>
            <td>Use <code>imgsz=1280</code>, enable mosaic augmentation</td>
          </tr>
          <tr>
            <td>High FP confidence</td>
            <td>Model is overconfident on wrong detections</td>
            <td>More diverse training data, raise confidence threshold</td>
          </tr>
          <tr>
            <td>Low small-object recall</td>
            <td>Drones far away are missed</td>
            <td>Synthetic data with small-scale objects, <code>imgsz=1280</code></td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>

</div>

<div class="footer">
  <div>UAV Drone Detection · YOLOv8 · Evaluation Report</div>
  <div>Precision <span>{stats['precision']:.3f}</span> · Recall <span>{stats['recall']:.3f}</span> · F1 <span>{stats['f1']:.3f}</span></div>
</div>

</body>
</html>"""

    report_path = os.path.join(OUT_DIR, "evaluation_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path

if __name__ == "__main__":
    setup()

    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Running evaluation...")
    stats = run_evaluation(model)

    print("Generating charts...")
    chart_paths = {
        "confusion":  chart_confusion(stats),
        "prf":        chart_prf(stats),
        "conf_dist":  chart_conf_dist(stats),
        "size_recall": chart_size_recall(stats),
        "img_recall": chart_image_recall_hist(stats),
    }

    print("Saving sample images...")
    sample_paths = save_sample_images(stats)

    print("Writing report...")
    report_path = write_report(stats, chart_paths, sample_paths)

    print(f"\n{'='*50}")
    print(f"  Precision : {stats['precision']:.3f}")
    print(f"  Recall    : {stats['recall']:.3f}")
    print(f"  F1        : {stats['f1']:.3f}")
    print(f"  TP / FP / FN : {stats['TP']} / {stats['FP']} / {stats['FN']}")
    print(f"{'='*50}")
    print(f"\nReport saved to: {report_path}")