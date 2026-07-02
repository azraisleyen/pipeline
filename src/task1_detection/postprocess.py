from __future__ import annotations

from src.common.bbox_ops import clamp_bbox, iou
from src.common.constants import LANDING_NOT_APPLICABLE, MOTION_NOT_APPLICABLE


def make_object(cls, bbox, landing_status=LANDING_NOT_APPLICABLE, motion_status=MOTION_NOT_APPLICABLE, score: float | None = None):
    x1, y1, x2, y2 = clamp_bbox(bbox)
    obj = {
        "cls": str(cls),
        "landing_status": str(landing_status),
        "motion_status": str(motion_status),
        "top_left_x": x1,
        "top_left_y": y1,
        "bottom_right_x": x2,
        "bottom_right_y": y2,
    }
    if score is not None:
        obj["_score"] = float(score)
    return obj


def strip_internal_fields(obj):
    return {k: v for k, v in obj.items() if not k.startswith("_")}


def nms_objects(objects, iou_threshold=0.75):
    ordered = sorted(objects, key=lambda obj: float(obj.get("_score", 0.0)), reverse=True)
    kept = []
    for obj in ordered:
        b = (obj["top_left_x"], obj["top_left_y"], obj["bottom_right_x"], obj["bottom_right_y"])
        suppress = False
        for kept_obj in kept:
            if obj.get("cls") != kept_obj.get("cls"):
                continue
            kb = (kept_obj["top_left_x"], kept_obj["top_left_y"], kept_obj["bottom_right_x"], kept_obj["bottom_right_y"])
            if iou(b, kb) >= iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(obj)
    return [strip_internal_fields(obj) for obj in kept]
