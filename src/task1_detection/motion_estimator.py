from __future__ import annotations

from dataclasses import dataclass

from src.common.bbox_ops import center, iou


@dataclass
class MotionTrack:
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    frame_id: int
    track_id: int
    matched: bool = False


class VehicleMotionEstimator:
    """Lightweight multi-object vehicle motion estimator.

    It keeps internal-only tracks and associates detections by IoU/center distance.
    Output remains the official motion_status string only.
    """

    def __init__(self, threshold_px=12.0, max_lost_frames=5, association_iou=0.15, association_distance_px=96.0):
        self.threshold = float(threshold_px)
        self.max_lost = int(max_lost_frames)
        self.association_iou = float(association_iou)
        self.association_distance_px = float(association_distance_px)
        self.tracks: dict[int, MotionTrack] = {}
        self._next_id = 1
        self._last_frame_id: int | None = None

    def begin_frame(self, frame_id: int) -> None:
        if self._last_frame_id != frame_id:
            for track in self.tracks.values():
                track.matched = False
            self._last_frame_id = frame_id
            self._prune(frame_id)

    def estimate(self, bbox, frame_id):
        self.begin_frame(frame_id)
        box = tuple(int(round(float(v))) for v in bbox)
        c = center(box)
        track = self._match(box, c, frame_id)
        if track is None:
            self.tracks[self._next_id] = MotionTrack(c, box, frame_id, self._next_id, matched=True)
            self._next_id += 1
            return "0"
        dist = ((c[0] - track.center[0]) ** 2 + (c[1] - track.center[1]) ** 2) ** 0.5
        track.center = c
        track.bbox = box
        track.frame_id = frame_id
        track.matched = True
        return "1" if dist >= self.threshold else "0"

    def _match(self, bbox, c, frame_id):
        best_track = None
        best_score = -1.0
        for track in self.tracks.values():
            if track.matched or frame_id - track.frame_id > self.max_lost:
                continue
            overlap = iou(bbox, track.bbox)
            dist = ((c[0] - track.center[0]) ** 2 + (c[1] - track.center[1]) ** 2) ** 0.5
            if overlap < self.association_iou and dist > self.association_distance_px:
                continue
            score = overlap - (dist / max(self.association_distance_px, 1.0)) * 0.1
            if score > best_score:
                best_score = score
                best_track = track
        return best_track

    def _prune(self, frame_id: int) -> None:
        stale = [track_id for track_id, track in self.tracks.items() if frame_id - track.frame_id > self.max_lost]
        for track_id in stale:
            self.tracks.pop(track_id, None)

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
        self._last_frame_id = None
