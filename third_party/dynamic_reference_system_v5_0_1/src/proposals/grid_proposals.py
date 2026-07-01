
from typing import List, Dict
from collections import defaultdict


class GridProposalGenerator:
    def __init__(
        self,
        enabled=True,
        window_sizes=None,
        stride_ratio=0.50,
        aspect_ratios=None,
        max_grid_candidates=120,
        balanced_sampling=True,
        spatial_bins_x=4,
        spatial_bins_y=4,
    ):
        self.enabled = bool(enabled)
        self.window_sizes = list(window_sizes or [48, 64, 96, 128, 192, 256, 384])
        self.stride_ratio = float(stride_ratio)
        self.aspect_ratios = list(aspect_ratios or [0.75, 1.0, 1.33, 1.78])
        self.max_grid_candidates = int(max_grid_candidates)
        self.balanced_sampling = bool(balanced_sampling)
        self.spatial_bins_x = int(spatial_bins_x)
        self.spatial_bins_y = int(spatial_bins_y)

    def _make_candidate(self, frame_name, frame_index, rank, x1, y1, x2, y2, base_size, aspect_ratio):
        return {
            "candidate_id": f"{frame_name}__grid__{rank}",
            "frame_id": int(frame_index),
            "frame_name": frame_name,
            "candidate_rank": int(rank),
            "source_type": "grid",
            "target_reference_id": None,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "proposal_score": 0.0,
            "detector_conf": 0.0,
            "grid_base_size": int(base_size),
            "grid_aspect_ratio": float(aspect_ratio),
        }

    def _bin_key(self, candidate, image_w, image_h):
        x1, y1, x2, y2 = candidate["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        bx = int(min(self.spatial_bins_x - 1, max(0, cx / max(1, image_w) * self.spatial_bins_x)))
        by = int(min(self.spatial_bins_y - 1, max(0, cy / max(1, image_h) * self.spatial_bins_y)))

        return by, bx

    def _balanced_sample(self, proposals, image_w, image_h):
        if len(proposals) <= self.max_grid_candidates:
            return proposals

        bins = defaultdict(list)
        for p in proposals:
            bins[self._bin_key(p, image_w, image_h)].append(p)

        for key in bins:
            bins[key] = sorted(
                bins[key],
                key=lambda p: (
                    p.get("grid_base_size", 0),
                    p["bbox"][1],
                    p["bbox"][0],
                )
            )

        ordered_keys = sorted(bins.keys())
        selected = []

        while len(selected) < self.max_grid_candidates:
            added = False

            for key in ordered_keys:
                if bins[key]:
                    selected.append(bins[key].pop(0))
                    added = True

                    if len(selected) >= self.max_grid_candidates:
                        break

            if not added:
                break

        for new_rank, p in enumerate(selected):
            p["candidate_rank"] = int(new_rank)
            p["candidate_id"] = f'{p["frame_name"]}__grid__{new_rank}'

        return selected

    def generate(self, image_w: int, image_h: int, frame_name: str, frame_index: int) -> List[Dict]:
        if not self.enabled:
            return []

        proposals = []
        seen = set()
        rank = 0

        for base_size in self.window_sizes:
            for ar in self.aspect_ratios:
                ar = float(ar)

                if ar <= 0:
                    continue

                w = int(round(base_size * (ar ** 0.5)))
                h = int(round(base_size / (ar ** 0.5)))

                w = max(8, w)
                h = max(8, h)

                if w >= image_w or h >= image_h:
                    continue

                stride_x = max(8, int(round(w * self.stride_ratio)))
                stride_y = max(8, int(round(h * self.stride_ratio)))

                y_positions = list(range(0, max(1, image_h - h + 1), stride_y))
                x_positions = list(range(0, max(1, image_w - w + 1), stride_x))

                if len(y_positions) == 0 or y_positions[-1] != image_h - h:
                    y_positions.append(max(0, image_h - h))

                if len(x_positions) == 0 or x_positions[-1] != image_w - w:
                    x_positions.append(max(0, image_w - w))

                for y1 in y_positions:
                    for x1 in x_positions:
                        x2 = min(image_w - 1, x1 + w)
                        y2 = min(image_h - 1, y1 + h)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        key = (int(x1), int(y1), int(x2), int(y2))
                        if key in seen:
                            continue

                        seen.add(key)

                        proposals.append(
                            self._make_candidate(
                                frame_name=frame_name,
                                frame_index=frame_index,
                                rank=rank,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                base_size=base_size,
                                aspect_ratio=ar,
                            )
                        )
                        rank += 1

        if self.balanced_sampling:
            return self._balanced_sample(proposals, image_w=image_w, image_h=image_h)

        return proposals[: self.max_grid_candidates]
