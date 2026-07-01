
import math

from ..core.utils import compute_iou, clip_bbox_xyxy


class TemporalManager:
    def __init__(self, alpha=0.70):
        self.alpha = float(alpha)

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _size(box):
        x1, y1, x2, y2 = box
        return max(1.0, x2 - x1), max(1.0, y2 - y1)

    def smooth_bbox(self, prev_bbox, new_bbox):
        if prev_bbox is None:
            return [float(v) for v in new_bbox]

        smoothed = []
        for i in range(4):
            val = self.alpha * float(new_bbox[i]) + (1.0 - self.alpha) * float(prev_bbox[i])
            smoothed.append(val)

        return smoothed

    def center_proximity_score(self, prev_bbox, new_bbox):
        if prev_bbox is None:
            return 0.0

        px, py = self._center(prev_bbox)
        nx, ny = self._center(new_bbox)

        pw, ph = self._size(prev_bbox)
        norm = max(math.sqrt(pw * pw + ph * ph), 1.0)

        dist = math.sqrt((px - nx) ** 2 + (py - ny) ** 2)
        score = 1.0 - min(1.0, dist / (2.0 * norm))
        return max(0.0, min(1.0, score))

    def size_similarity_score(self, prev_bbox, new_bbox):
        if prev_bbox is None:
            return 0.0

        pw, ph = self._size(prev_bbox)
        nw, nh = self._size(new_bbox)

        area_prev = pw * ph
        area_new = nw * nh

        ratio = min(area_prev, area_new) / max(area_prev, area_new, 1.0)
        return max(0.0, min(1.0, ratio))

    def temporal_score(self, prev_bbox, new_bbox):
        if prev_bbox is None:
            return 0.0

        iou_score = compute_iou(prev_bbox, new_bbox)
        center_score = self.center_proximity_score(prev_bbox, new_bbox)
        size_score = self.size_similarity_score(prev_bbox, new_bbox)

        score = 0.50 * iou_score + 0.25 * center_score + 0.25 * size_score
        return float(max(0.0, min(1.0, score)))

    def generate_local_proposals(
        self,
        prev_bbox,
        image_w,
        image_h,
        search_radius_factor=0.20,
        scale_factors=(1.00,),
        offsets=(-1, 0, 1),
    ):
        if prev_bbox is None:
            return []

        x1, y1, x2, y2 = [float(v) for v in prev_bbox]
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        dx_base = w * float(search_radius_factor)
        dy_base = h * float(search_radius_factor)

        proposals = []
        seen = set()

        for scale in scale_factors:
            sw = max(2.0, w * float(scale))
            sh = max(2.0, h * float(scale))

            for ox in offsets:
                for oy in offsets:
                    pcx = cx + float(ox) * dx_base
                    pcy = cy + float(oy) * dy_base

                    box = [
                        pcx - sw / 2.0,
                        pcy - sh / 2.0,
                        pcx + sw / 2.0,
                        pcy + sh / 2.0,
                    ]

                    box = clip_bbox_xyxy(box, image_w, image_h)
                    bx1, by1, bx2, by2 = box

                    if bx2 <= bx1 or by2 <= by1:
                        continue

                    key = tuple(int(v) for v in box)
                    if key in seen:
                        continue

                    seen.add(key)
                    proposals.append(box)

        return proposals
