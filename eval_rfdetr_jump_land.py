import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from utils.io import read_yaml, read_json, write_json


def inference_one_video(model, video_frame_dir, threshold=0.25, batch_size=16):
    print("Infering on video:", video_frame_dir)
    frames = sorted(glob.glob(str(video_frame_dir / "*jpg")))

    all_results = []
    # Process frames in batches
    for i in tqdm(range(0, len(frames), batch_size)):
        batch_frames = frames[i : i + batch_size]
        results = model.predict(batch_frames, threshold=threshold)
        all_results.extend(results)

    return all_results


def load_model(checkpoint, batch_size=16):
    from rfdetr import RFDETRBase

    model = RFDETRBase(pretrain_weights=checkpoint)
    # model.optimize_for_inference(batch_size=batch_size)
    return model


def _iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1 + 1)
    inter_h = max(0.0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1 + 1) * max(0.0, ay2 - ay1 + 1)
    area_b = max(0.0, bx2 - bx1 + 1) * max(0.0, by2 - by1 + 1)

    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def _temporal_nms(detections, tolerance_t=3, iou_threshold=0.5):
    if len(detections) == 0:
        return []

    order = sorted(range(len(detections)), key=lambda i: detections[i][5], reverse=True)
    keep = []
    suppressed = np.zeros(len(detections), dtype=bool)

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        ti, box_i, _ = detections[i][0], detections[i][1:5], detections[i][5]
        for j in order:
            if i == j or suppressed[j]:
                continue
            tj, box_j, _ = detections[j][0], detections[j][1:5], detections[j][5]
            if abs(tj - ti) <= tolerance_t:
                if _iou_xyxy(box_i, box_j) >= iou_threshold:
                    suppressed[j] = True

    return [detections[i] for i in keep]


def postprocess_results(
    sv_results,
    classes,
    background_class=0,
    use_temporal_nms=True,
    nms_tolerance_t=3,
    nms_iou_threshold=0.5,
):
    pred_events = {}

    by_class = {cls_id: [] for cls_id in range(len(classes))}

    # Aggregate raw detections
    for frame_idx, result in enumerate(sv_results):
        xyxy = np.asarray(result.xyxy)
        class_id = np.asarray(result.class_id)
        confidence = np.asarray(result.confidence)

        if xyxy.size == 0:
            continue

        for obj_idx, _cls in enumerate(class_id):
            if _cls == background_class:
                continue
            x1, y1, x2, y2 = map(float, xyxy[obj_idx])
            score = float(confidence[obj_idx])
            by_class[int(_cls)].append((frame_idx, x1, y1, x2, y2, score))

    # Temporal NMS per class (optional)
    for _cls, dets in by_class.items():
        if _cls == background_class or len(dets) == 0:
            continue
        if use_temporal_nms:
            kept = _temporal_nms(
                dets, tolerance_t=nms_tolerance_t, iou_threshold=nms_iou_threshold
            )
        else:
            kept = dets

        for frame_idx, x1, y1, x2, y2, score in kept:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            event = classes[_cls]
            pred_events[event] = pred_events.get(event, [])
            pred_events[event].append(
                (
                    frame_idx,
                    float(center_x),
                    float(center_y),
                    float(score),
                )
            )
    return pred_events


def get_frame_idx(file_name):
    return int(file_name[:-4].split("_")[-1])


def parse_gt(annotations, videos_list, classes, background_class=0):

    events_by_image_id = {
        image["id"]: [
            obj
            for obj in annotations["annotations"]
            if obj["category_id"] != background_class and obj["image_id"] == image["id"]
        ]
        for image in annotations["images"]
    }

    frames_by_video_name = {
        video_name: natsorted(
            [
                image
                for image in annotations["images"]
                if video_name in image["file_name"]
            ],
            key=lambda image: image["file_name"],
        )
        for video_name in videos_list
    }

    truths = {
        video_name: {
            _cls: [
                (
                    get_frame_idx(image["file_name"]),
                    event["bbox"][0] + event["bbox"][2] / 2,
                    event["bbox"][1] + event["bbox"][3] / 2,
                )
                for image in frames_list
                for event in events_by_image_id[image["id"]]
                if event["category_id"] == _clsid
            ]
            for _clsid, _cls in enumerate(classes)
        }
        for video_name, frames_list in frames_by_video_name.items()
    }

    return truths


# ==========================================================
# Matching (1D time, many-to-many safe)
# ==========================================================
def match_events_time_only(
    gt_list, pred_list, tolerance=2  # [(t,x,y)]  # [(t,x,y,score)]
):
    """
    Greedy one-to-one matching within ±tolerance frames.
    Allows multiple events at same frame.
    """
    gt_sorted = sorted(gt_list, key=lambda x: x[0])
    pred_sorted = sorted(
        pred_list, key=lambda x: (x[0], -(x[3] if x[3] is not None else -1e9))
    )

    used_pred = np.zeros(len(pred_sorted), dtype=bool)
    matched = []

    for gt in gt_sorted:
        gt_t = gt[0]
        best_j, best_dt = None, None

        for j, pr in enumerate(pred_sorted):
            if used_pred[j]:
                continue
            dt = abs(pr[0] - gt_t)
            if dt <= tolerance:
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_j = j

        if best_j is not None:
            used_pred[best_j] = True
            matched.append((gt, pred_sorted[best_j]))

    unmatched_gt = [g for g in gt_sorted if all(g is not m[0] for m in matched)]
    unmatched_pred = [p for j, p in enumerate(pred_sorted) if not used_pred[j]]

    return matched, unmatched_gt, unmatched_pred


# ==========================================================
# Error statistics
# ==========================================================
def summarize_matches(matched):
    if len(matched) == 0:
        return {"mean_abs_dt": np.nan, "mean_xy_dist": np.nan}

    dts, dxy = [], []

    for gt, pr in matched:
        gt_t, gt_x, gt_y = gt
        pr_t, pr_x, pr_y, _ = pr

        dts.append(abs(pr_t - gt_t))

        if gt_x is not None and pr_x is not None:
            dxy.append(np.sqrt((pr_x - gt_x) ** 2 + (pr_y - gt_y) ** 2))

    return {
        "mean_abs_dt": float(np.mean(dts)),
        "mean_xy_dist": float(np.mean(dxy)) if len(dxy) > 0 else np.nan,
    }


# ==========================================================
# Main validation
# ==========================================================
def valid_jump_land(
    preds: dict,
    truths: dict,
    tolerance: int = 2,
    score_thresh: float = 0.18,  # ⭐ REQUIRED
):

    # for fname in natsorted(os.listdir(results_dir)):
    #     if not fname.endswith(".csv"):
    #         continue

    assert set(preds.keys()).issubset(
        set(truths.keys())
    ), "All Pred videos should be included in Ground Truths"

    totals = {
        "jump": {"tp": 0, "fp": 0, "fn": 0, "matched": []},
        "land": {"tp": 0, "fp": 0, "fn": 0, "matched": []},
    }

    for video_name, predictions in preds.items():
        # data = collect_gt_pred_from_jump_land_csv(os.path.join(results_dir, fname))

        for ev in ["jump", "land"]:
            # gt = data[ev]["gt"]
            gt = truths[video_name][ev]

            # --------------------------------------------------
            # 🔴 FILTER PREDICTIONS BY SCORE
            # --------------------------------------------------
            pred = [
                # p for p in data[ev]["pred"]
                p
                for p in predictions[ev]
                if (p[3] is None) or (p[3] >= score_thresh)
            ]

            matched, un_gt, un_pr = match_events_time_only(
                gt_list=gt, pred_list=pred, tolerance=tolerance
            )

            totals[ev]["tp"] += len(matched)
            totals[ev]["fn"] += len(un_gt)
            totals[ev]["fp"] += len(un_pr)
            totals[ev]["matched"].extend(matched)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    results = {}

    for ev in ["jump", "land"]:
        tp, fp, fn = totals[ev]["tp"], totals[ev]["fp"], totals[ev]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        errs = summarize_matches(totals[ev]["matched"])

        results[ev] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": f"{prec:0.3f}",
            "recall": f"{rec:0.3f}",
            "f1": f"{f1:0.3f}",
            **errs,
        }

    # overall
    tp = results["jump"]["tp"] + results["land"]["tp"]
    fp = results["jump"]["fp"] + results["land"]["fp"]
    fn = results["jump"]["fn"] + results["land"]["fn"]

    overall_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    overall_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_prec * overall_rec / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0.0
    )

    results["overall"] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": f"{overall_prec:0.3f}",
        "recall": f"{overall_rec:0.3f}",
        "f1": f"{overall_f1:0.3f}",
    }

    return results


if __name__ == "__main__":
    CLASSES = ["player", "jump", "land"]
    tolerance = 4
    score_thresh = 0.5
    checkpoint_path = "finetune/RFDETRBase/checkpoint_best_total.pth"
    annotation_file_path = "data/VNL_500Videos_RFDETR/test/_annotations.coco.json"
    test_videos_list = read_yaml("data/VNL_500Videos_RFDETR/split_videos.yaml")["test"]

    truths = read_json(annotation_file_path)
    truths = parse_gt(truths, test_videos_list, classes=CLASSES, background_class=0)

    root_data_dir = Path("data/VNL_500Videos")
    eval_dir = Path("./eval/")
    eval_dir.mkdir(exist_ok=True)

    # load model
    model = load_model(checkpoint_path)
    preds = {}

    for idx, video_name in enumerate(test_videos_list):
        preds_raw = inference_one_video(
            model, root_data_dir / video_name, batch_size=32
        )
        preds[video_name] = postprocess_results(preds_raw, classes=CLASSES)
        print(
            f"Finished {idx+1} / {len(test_videos_list)} ({(idx+1)/len(test_videos_list):.2%})"
        )

    write_json(eval_dir / "preds.json", preds)
    write_json(eval_dir / "truths.json", truths)

    metrics = valid_jump_land(
        preds=preds,
        truths=truths,
        tolerance=tolerance,
        score_thresh=score_thresh,
    )

    print("\n========== Jump–Land Event Spotting Evaluation ==========")
    print(f"Score threshold: {score_thresh} | Tolerance: ±{tolerance} frames")

    for k, v in metrics.items():
        print(f"\n[{k.upper()}]")
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")
