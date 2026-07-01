
class ScoreFusion:
    def __init__(
        self,
        appearance_weight=0.80,
        geometry_weight=0.15,
        temporal_weight=0.05,
        mellin_weight=0.00,
        min_final_score=0.30,
        min_final_margin=0.00,
        min_final_score_with_geometry=0.24,
        strong_visual_accept_score=0.38,
        strong_visual_min_margin=0.00,
        yolo_source_bonus=0.02,
        grid_source_bonus=0.00,
        local_source_penalty=0.04,
    ):
        self.appearance_weight = float(appearance_weight)
        self.geometry_weight = float(geometry_weight)
        self.temporal_weight = float(temporal_weight)
        self.mellin_weight = float(mellin_weight)

        self.min_final_score = float(min_final_score)
        self.min_final_margin = float(min_final_margin)
        self.min_final_score_with_geometry = float(min_final_score_with_geometry)

        self.strong_visual_accept_score = float(strong_visual_accept_score)
        self.strong_visual_min_margin = float(strong_visual_min_margin)

        self.yolo_source_bonus = float(yolo_source_bonus)
        self.grid_source_bonus = float(grid_source_bonus)
        self.local_source_penalty = float(local_source_penalty)

    def source_adjustment(self, source_type):
        if source_type in {"yolo", "detector"}:
            return self.yolo_source_bonus
        if source_type == "grid":
            return self.grid_source_bonus
        if source_type in {"local", "local_search"}:
            return -self.local_source_penalty
        return 0.0

    def fuse(self, row):
        appearance = float(row.get("appearance_score", 0.0))
        geometry = float(row.get("geometry_score", 0.0))
        temporal = float(row.get("temporal_score", 0.0))
        mellin = float(row.get("mellin_score", 0.0))

        final_score = (
            self.appearance_weight * appearance
            + self.geometry_weight * geometry
            + self.temporal_weight * temporal
            + self.mellin_weight * mellin
        )

        compare_score = final_score + self.source_adjustment(row.get("source_type", ""))

        return float(final_score), float(compare_score)

    def decide(self, row):
        final_score = float(row.get("final_score", 0.0))
        margin = float(row.get("margin", 0.0))
        appearance = float(row.get("appearance_score", 0.0))
        geometry_pass = bool(row.get("geometry_pass", False))

        if (
            appearance >= self.strong_visual_accept_score
            and margin >= self.strong_visual_min_margin
        ):
            return "accept_visual_strong"

        if (
            geometry_pass
            and final_score >= self.min_final_score_with_geometry
            and margin >= self.min_final_margin
        ):
            return "accept_geom_supported"

        if final_score < self.min_final_score:
            return "reject_low_final_score"

        if margin < self.min_final_margin:
            return "reject_low_final_margin"

        return "accept_fused"
