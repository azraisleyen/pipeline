
from typing import List, Dict


def _area(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    union = _area(a) + _area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


class ProposalFusion:
    def __init__(
        self,
        max_total_candidates_per_frame=160,
        nms_iou_threshold=0.65,
        min_bbox_area=32,
        max_bbox_area_ratio=0.70,
        min_aspect_ratio=0.12,
        max_aspect_ratio=7.00,
    ):
        self.max_total_candidates_per_frame = int(max_total_candidates_per_frame)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.min_bbox_area = float(min_bbox_area)
        self.max_bbox_area_ratio = float(max_bbox_area_ratio)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)

        self.source_priority = {
            "yolo": 4,
            "detector": 4,
            "local_search": 3,
            "local": 3,
            "grid": 2,
            "contour": 1,
            "sam": 1,
        }

    def _filter_reason(self, proposal, image_w, image_h):
        x1, y1, x2, y2 = [float(v) for v in proposal["bbox"]]

        x1 = max(0.0, min(float(image_w - 1), x1))
        y1 = max(0.0, min(float(image_h - 1), y1))
        x2 = max(0.0, min(float(image_w - 1), x2))
        y2 = max(0.0, min(float(image_h - 1), y2))

        proposal["bbox"] = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

        if x2 <= x1 or y2 <= y1:
            return "invalid_bbox"

        w = x2 - x1
        h = y2 - y1
        area = w * h
        image_area = max(1.0, float(image_w * image_h))

        if area < self.min_bbox_area:
            return "too_small"

        if area / image_area > self.max_bbox_area_ratio:
            return "too_large"

        aspect = w / max(h, 1.0)
        if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
            return "bad_aspect"

        return "keep"

    def _sort_key(self, p):
        source = p.get("source_type", "unknown")
        priority = self.source_priority.get(source, 0)
        score = float(p.get("proposal_score", p.get("detector_conf", 0.0)))
        area = _area(p["bbox"])
        return (priority, score, -area)

    def fuse(self, proposals: List[Dict], image_w: int, image_h: int):
        proposal_logs = []
        valid = []

        for p in proposals:
            p = dict(p)
            reason = self._filter_reason(p, image_w=image_w, image_h=image_h)

            log_row = {
                "frame_id": p.get("frame_id"),
                "frame_name": p.get("frame_name"),
                "candidate_id": p.get("candidate_id"),
                "source_type": p.get("source_type"),
                "target_reference_id": p.get("target_reference_id"),
                "bbox_x1": p["bbox"][0],
                "bbox_y1": p["bbox"][1],
                "bbox_x2": p["bbox"][2],
                "bbox_y2": p["bbox"][3],
                "proposal_score": float(p.get("proposal_score", 0.0)),
                "detector_conf": float(p.get("detector_conf", 0.0)),
                "kept_after_filter": reason == "keep",
                "filter_reason": reason,
                "kept_after_nms": False,
            }

            if reason == "keep":
                valid.append(p)

            proposal_logs.append(log_row)

        valid = sorted(valid, key=self._sort_key, reverse=True)

        kept = []
        kept_ids = set()

        for p in valid:
            suppress = False
            for k in kept:
                if _iou(p["bbox"], k["bbox"]) >= self.nms_iou_threshold:
                    suppress = True
                    break

            if not suppress:
                kept.append(p)
                kept_ids.add(p["candidate_id"])

            if len(kept) >= self.max_total_candidates_per_frame:
                break

        for row in proposal_logs:
            if row["candidate_id"] in kept_ids:
                row["kept_after_nms"] = True

        return kept, proposal_logs
