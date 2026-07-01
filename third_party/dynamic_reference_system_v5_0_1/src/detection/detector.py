
from ultralytics import YOLO

from ..core.utils import clip_bbox_xyxy, bbox_area, bbox_aspect_ratio


class YOLODetector:
    def __init__(
        self,
        model_path,
        device="cpu",
        conf=0.04,
        iou=0.50,
        max_det=100,
        top_k_candidates=60,
        min_bbox_area=32,
        max_bbox_area_ratio=0.70,
        min_aspect_ratio=0.12,
        max_aspect_ratio=7.00,
    ):
        self.model = YOLO(str(model_path))
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.top_k_candidates = int(top_k_candidates)
        self.min_bbox_area = int(min_bbox_area)
        self.max_bbox_area_ratio = float(max_bbox_area_ratio)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)

    def detect(self, image_bgr):
        image_h, image_w = image_bgr.shape[:2]
        image_area = image_h * image_w

        result = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )[0]

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = clip_bbox_xyxy([x1, y1, x2, y2], image_w, image_h)

            area = bbox_area(bbox)
            if area < self.min_bbox_area:
                continue

            area_ratio = float(area) / float(max(image_area, 1))
            if area_ratio > self.max_bbox_area_ratio:
                continue

            aspect_ratio = bbox_aspect_ratio(bbox)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            detections.append(
                {
                    "bbox_xyxy": bbox,
                    "conf": float(box.conf.item()),
                    "area": area,
                    "area_ratio": area_ratio,
                    "aspect_ratio": aspect_ratio,
                    "source": "detector",
                }
            )

        detections.sort(key=lambda d: d["conf"], reverse=True)
        detections = detections[: self.top_k_candidates]

        for rank, det in enumerate(detections):
            det["candidate_rank"] = rank

        return detections
