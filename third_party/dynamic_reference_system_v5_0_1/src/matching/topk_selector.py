
from collections import defaultdict


class TopKSelector:
    def __init__(
        self,
        top_k_per_reference=15,
        top_k_for_geometry=3,
        min_appearance_for_geometry=0.18,
    ):
        self.top_k_per_reference = int(top_k_per_reference)
        self.top_k_for_geometry = int(top_k_for_geometry)
        self.min_appearance_for_geometry = float(min_appearance_for_geometry)

    def select(self, scored_pairs):
        by_ref = defaultdict(list)

        for row in scored_pairs:
            by_ref[row["reference_id"]].append(row)

        topk_by_ref = {}
        geometry_by_ref = {}

        for ref_id, rows in by_ref.items():
            sorted_rows = sorted(
                rows,
                key=lambda x: float(x.get("appearance_score", 0.0)),
                reverse=True,
            )

            for rank, row in enumerate(sorted_rows):
                row["rank_for_reference"] = int(rank + 1)

            topk_rows = sorted_rows[: self.top_k_per_reference]

            geometry_candidates = [
                row for row in sorted_rows
                if float(row.get("appearance_score", 0.0)) >= self.min_appearance_for_geometry
            ]

            if len(geometry_candidates) == 0 and len(sorted_rows) > 0:
                geometry_candidates = [sorted_rows[0]]

            geometry_rows = geometry_candidates[: self.top_k_for_geometry]

            topk_by_ref[ref_id] = topk_rows
            geometry_by_ref[ref_id] = geometry_rows

        return topk_by_ref, geometry_by_ref
