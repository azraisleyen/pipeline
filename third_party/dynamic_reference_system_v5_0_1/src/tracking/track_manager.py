
from .track_state import Track
from ..temporal.temporal import TemporalManager


class TrackManager:
    def __init__(self, config):
        self.enabled = bool(config.tracking.get("enabled", True))
        self.max_lost = int(config.tracking.get("max_lost", 3))
        self.smoothing_alpha = float(config.tracking.get("smoothing_alpha", 0.70))

        self.local_search_enabled = bool(config.tracking.get("local_search_enabled", True))
        self.local_search_radius_factor = float(config.tracking.get("local_search_radius_factor", 0.20))
        self.local_search_scale_factors = list(config.tracking.get("local_search_scale_factors", [1.00]))
        self.local_search_offsets = list(config.tracking.get("local_search_offsets", [-1, 0, 1]))

        self.local_appearance_weight = float(config.tracking.get("local_appearance_weight", 0.55))
        self.local_temporal_weight = float(config.tracking.get("local_temporal_weight", 0.25))
        self.local_margin_weight = float(config.tracking.get("local_margin_weight", 0.20))

        self.local_min_similarity = float(config.tracking.get("local_min_similarity", 0.28))
        self.local_min_margin = float(config.tracking.get("local_min_margin", 0.00))
        self.local_min_temporal_score = float(config.tracking.get("local_min_temporal_score", 0.20))
        self.local_min_final_score = float(config.tracking.get("local_min_final_score", 0.32))

        self.local_early_stop_enabled = bool(config.tracking.get("local_early_stop_enabled", True))
        self.local_early_stop_similarity = float(config.tracking.get("local_early_stop_similarity", 0.42))
        self.local_early_stop_temporal_score = float(config.tracking.get("local_early_stop_temporal_score", 0.55))

        self.force_detector_refresh_every_n_frames = int(config.tracking.get("force_detector_refresh_every_n_frames", 3))
        self.max_local_only_streak = int(config.tracking.get("max_local_only_streak", 2))

        self.prefer_detector_when_available = bool(config.tracking.get("prefer_detector_when_available", True))
        self.detector_preference_delta = float(config.tracking.get("detector_preference_delta", 0.02))

        self.birth_min_score = float(config.tracking.get("birth_min_score", 0.30))
        self.birth_min_margin = float(config.tracking.get("birth_min_margin", 0.00))
        self.birth_min_detector_conf = float(config.tracking.get("birth_min_detector_conf", 0.00))

        self.tracked_min_detector_score = float(config.tracking.get("tracked_min_detector_score", 0.25))
        self.tracked_min_detector_margin = float(config.tracking.get("tracked_min_detector_margin", 0.00))

        self.lost_min_detector_score = float(config.tracking.get("lost_min_detector_score", 0.28))
        self.lost_min_detector_margin = float(config.tracking.get("lost_min_detector_margin", 0.00))

        self.cross_reference_iou_threshold = float(config.tracking.get("cross_reference_iou_threshold", 0.70))

        self.temporal = TemporalManager(alpha=self.smoothing_alpha)

        self.tracks = {}
        self.next_track_id = 0

    def initialize_references(self, reference_ids):
        for ref_id in sorted(reference_ids):
            if ref_id not in self.tracks:
                self.tracks[ref_id] = Track(self.next_track_id, ref_id)
                self.next_track_id += 1

    def get_track(self, reference_id):
        return self.tracks[reference_id]

    def get_all_tracks(self):
        return [self.tracks[k] for k in sorted(self.tracks.keys())]

    def get_searchable_tracks(self):
        out = []
        for track in self.get_all_tracks():
            if track.state in {"TRACKING", "LOST"} and track.bbox is not None:
                out.append(track)
        return out

    def snapshot_states(self):
        return {ref_id: track.to_dict() for ref_id, track in self.tracks.items()}

    def _is_global_source(self, source):
        return source in {"detector", "yolo", "grid", "contour", "sam"}

    def update_with_candidate(
        self,
        reference_id,
        bbox,
        score,
        frame_index,
        source,
        appearance_score=0.0,
        detector_conf=0.0,
        temporal_score=0.0,
    ):
        track = self.get_track(reference_id)

        prev_bbox = track.bbox
        raw_bbox = [float(v) for v in bbox]

        if self._is_global_source(source):
            updated_bbox = raw_bbox
        else:
            updated_bbox = self.temporal.smooth_bbox(prev_bbox, raw_bbox)

        track.prev_bbox = prev_bbox
        track.raw_bbox = raw_bbox
        track.bbox = [float(v) for v in updated_bbox]

        track.state = "TRACKING"
        track.last_frame_index = int(frame_index)
        track.hits += 1
        track.lost_count = 0
        track.age += 1

        if self._is_global_source(source):
            track.local_only_streak = 0
            track.frames_since_detector = 0
        else:
            track.local_only_streak += 1
            track.frames_since_detector += 1

        track.last_score = float(score)
        track.last_appearance_score = float(appearance_score)
        track.last_detector_conf = float(detector_conf)
        track.last_temporal_score = float(temporal_score)
        track.last_source = str(source)

    def mark_unmatched_tracks(self, matched_reference_ids, frame_index):
        matched_reference_ids = set(matched_reference_ids)

        for track in self.get_all_tracks():
            if track.reference_id in matched_reference_ids:
                continue

            track.frames_since_detector += 1

            if track.state == "ABSENT":
                track.absent_count += 1
                continue

            track.lost_count += 1
            track.last_frame_index = int(frame_index)

            if track.lost_count > self.max_lost:
                track.state = "ABSENT"
                track.prev_bbox = track.bbox
                track.bbox = None
                track.raw_bbox = None
                track.last_source = "none"
                track.local_only_streak = 0
            else:
                track.state = "LOST"
