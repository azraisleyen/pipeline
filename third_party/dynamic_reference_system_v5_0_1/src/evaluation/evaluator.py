
from collections import defaultdict
import numpy as np

from ..core.utils import load_json, save_csv_rows, compute_iou


class Evaluator:
    def __init__(self):
        pass

    @staticmethod
    def _compute_ap(tp_flags, fp_flags, gt_total):
        if gt_total == 0:
            return 0.0

        tp_flags = np.asarray(tp_flags, dtype=np.float32)
        fp_flags = np.asarray(fp_flags, dtype=np.float32)

        tp_cum = np.cumsum(tp_flags)
        fp_cum = np.cumsum(fp_flags)

        recalls = tp_cum / max(float(gt_total), 1.0)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([1.0], precisions, [0.0]))

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

        return float(ap)

    def evaluate(
        self,
        gt_path,
        pred_path,
        threshold=0.5,
        per_reference_csv_path=None,
        fp_csv_path=None,
        fn_csv_path=None,
    ):
        gt = load_json(gt_path)
        pred = load_json(pred_path)

        gt_frame_order = {}
        gt_by_frame_ref = defaultdict(lambda: defaultdict(list))
        gt_totals = defaultdict(int)

        for frame_idx, frame in enumerate(gt["frames"]):
            frame_name = frame["frame_name"]
            gt_frame_order[frame_name] = frame_idx

            for obj in frame.get("objects", []):
                ref_id = obj["reference_id"]
                gt_by_frame_ref[frame_name][ref_id].append(
                    {
                        "bbox": obj["bbox"],
                        "matched": False,
                    }
                )
                gt_totals[ref_id] += 1

        preds_by_ref = defaultdict(list)

        for frame in pred.get("frames", []):
            frame_name = frame["frame_name"]
            for obj in frame.get("objects", []):
                preds_by_ref[obj["reference_id"]].append(
                    {
                        "frame_name": frame_name,
                        "bbox": obj["bbox"],
                        "score": float(obj.get("score", 0.0)),
                        "track_id": int(obj.get("track_id", -1)),
                    }
                )

        overall_tp = 0
        overall_fp = 0
        matched_track_memory = {}
        id_switch = 0

        per_reference_rows = []
        fp_records = []

        all_reference_ids = sorted(set(list(gt_totals.keys()) + list(preds_by_ref.keys())))

        for ref_id in all_reference_ids:
            pred_list = preds_by_ref.get(ref_id, [])
            pred_list = sorted(pred_list, key=lambda x: x["score"], reverse=True)

            gt_total = gt_totals.get(ref_id, 0)

            tp_flags = []
            fp_flags = []

            for pred_item in pred_list:
                frame_name = pred_item["frame_name"]
                pred_box = pred_item["bbox"]

                gt_candidates = gt_by_frame_ref[frame_name].get(ref_id, [])

                best_iou = 0.0
                best_idx = -1

                for idx, gt_item in enumerate(gt_candidates):
                    if gt_item["matched"]:
                        continue

                    iou = compute_iou(pred_box, gt_item["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_iou >= threshold and best_idx != -1:
                    gt_candidates[best_idx]["matched"] = True
                    tp_flags.append(1)
                    fp_flags.append(0)
                    overall_tp += 1

                    frame_idx = gt_frame_order.get(frame_name, -1)
                    track_id = pred_item["track_id"]

                    if track_id != -1:
                        prev = matched_track_memory.get(ref_id)
                        if prev is not None:
                            if prev["track_id"] != track_id and prev["frame_idx"] < frame_idx:
                                id_switch += 1

                        matched_track_memory[ref_id] = {
                            "track_id": track_id,
                            "frame_idx": frame_idx,
                        }

                else:
                    tp_flags.append(0)
                    fp_flags.append(1)
                    overall_fp += 1

                    fp_records.append(
                        {
                            "reference_id": ref_id,
                            "frame_name": frame_name,
                            "score": pred_item["score"],
                            "bbox": pred_box,
                            "best_iou": best_iou,
                        }
                    )

            ref_tp = int(sum(tp_flags))
            ref_fp = int(sum(fp_flags))
            ref_fn = int(gt_total - ref_tp)

            ref_precision = ref_tp / max(ref_tp + ref_fp, 1)
            ref_recall = ref_tp / max(gt_total, 1) if gt_total > 0 else 0.0
            ref_f1 = (
                2 * ref_precision * ref_recall / max(ref_precision + ref_recall, 1e-8)
                if (ref_precision + ref_recall) > 0
                else 0.0
            )
            ref_ap = self._compute_ap(tp_flags, fp_flags, gt_total)

            per_reference_rows.append(
                {
                    "reference_id": ref_id,
                    "gt_total": gt_total,
                    "pred_total": len(pred_list),
                    "tp": ref_tp,
                    "fp": ref_fp,
                    "fn": ref_fn,
                    "precision": ref_precision,
                    "recall": ref_recall,
                    "f1": ref_f1,
                    "ap50": ref_ap,
                }
            )

        fn_records = []
        for frame_name, ref_map in gt_by_frame_ref.items():
            for ref_id, gt_items in ref_map.items():
                for gt_item in gt_items:
                    if not gt_item["matched"]:
                        fn_records.append(
                            {
                                "reference_id": ref_id,
                                "frame_name": frame_name,
                                "bbox": gt_item["bbox"],
                            }
                        )

        total_gt = int(sum(gt_totals.values()))
        overall_fn = total_gt - overall_tp

        precision = overall_tp / max(overall_tp + overall_fp, 1)
        recall = overall_tp / max(total_gt, 1) if total_gt > 0 else 0.0
        f1 = (
            2 * precision * recall / max(precision + recall, 1e-8)
            if (precision + recall) > 0
            else 0.0
        )

        ap_values = [row["ap50"] for row in per_reference_rows if row["gt_total"] > 0]
        map50 = float(np.mean(ap_values)) if len(ap_values) > 0 else 0.0

        if per_reference_csv_path is not None:
            per_ref_fields = [
                "reference_id",
                "gt_total",
                "pred_total",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "ap50",
            ]
            save_csv_rows(per_reference_rows, per_reference_csv_path, fieldnames=per_ref_fields)

        if fp_csv_path is not None:
            fp_fields = ["reference_id", "frame_name", "score", "bbox", "best_iou"]
            save_csv_rows(fp_records, fp_csv_path, fieldnames=fp_fields)

        if fn_csv_path is not None:
            fn_fields = ["reference_id", "frame_name", "bbox"]
            save_csv_rows(fn_records, fn_csv_path, fieldnames=fn_fields)

        summary = {
            "total_gt": total_gt,
            "tp": overall_tp,
            "fp": overall_fp,
            "fn": overall_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": map50,
            "id_switch": id_switch,
            "num_references": len(per_reference_rows),
        }

        return summary
