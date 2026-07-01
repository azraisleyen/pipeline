
class Track:
    def __init__(self, track_id, reference_id):
        self.track_id = int(track_id)
        self.reference_id = str(reference_id)

        self.state = "ABSENT"

        self.bbox = None
        self.raw_bbox = None
        self.prev_bbox = None

        self.last_frame_index = -1
        self.hits = 0
        self.lost_count = 0
        self.absent_count = 0
        self.age = 0

        self.local_only_streak = 0
        self.frames_since_detector = 999999

        self.last_score = 0.0
        self.last_appearance_score = 0.0
        self.last_detector_conf = 0.0
        self.last_temporal_score = 0.0
        self.last_source = "none"

    def to_dict(self):
        return {
            "track_id": self.track_id,
            "reference_id": self.reference_id,
            "state": self.state,
            "bbox": self.bbox,
            "raw_bbox": self.raw_bbox,
            "prev_bbox": self.prev_bbox,
            "last_frame_index": self.last_frame_index,
            "hits": self.hits,
            "lost_count": self.lost_count,
            "absent_count": self.absent_count,
            "age": self.age,
            "local_only_streak": self.local_only_streak,
            "frames_since_detector": self.frames_since_detector,
            "last_score": self.last_score,
            "last_appearance_score": self.last_appearance_score,
            "last_detector_conf": self.last_detector_conf,
            "last_temporal_score": self.last_temporal_score,
            "last_source": self.last_source,
        }
