
class ReferenceMatcher:
    def __init__(
        self,
        tight_weight=0.75,
        context_weight=0.25,
        detector_conf_floor=0.04,
        min_similarity=0.27,
        min_margin=0.05,
        strong_accept_similarity=0.42,
        strong_accept_margin=0.10,
        ambiguity_score_lower=0.27,
        ambiguity_score_upper=0.40,
        ambiguity_margin_upper=0.10,
    ):
        self.tight_weight = float(tight_weight)
        self.context_weight = float(context_weight)
        self.detector_conf_floor = float(detector_conf_floor)
        self.min_similarity = float(min_similarity)
        self.min_margin = float(min_margin)

        self.strong_accept_similarity = float(strong_accept_similarity)
        self.strong_accept_margin = float(strong_accept_margin)

        self.ambiguity_score_lower = float(ambiguity_score_lower)
        self.ambiguity_score_upper = float(ambiguity_score_upper)
        self.ambiguity_margin_upper = float(ambiguity_margin_upper)

    def _score_reference(self, tight_embedding, context_embedding, reference_templates, embedder):
        tight_scores = []
        context_scores = []

        for template in reference_templates:
            ref_emb = template["embedding"]
            tight_scores.append(embedder.cosine(tight_embedding, ref_emb))
            context_scores.append(embedder.cosine(context_embedding, ref_emb))

        tight_best = max(tight_scores) if len(tight_scores) > 0 else 0.0
        context_best = max(context_scores) if len(context_scores) > 0 else 0.0

        appearance_score = (
            self.tight_weight * float(tight_best)
            + self.context_weight * float(context_best)
        )

        return {
            "tight_best_score": float(tight_best),
            "context_best_score": float(context_best),
            "appearance_score": float(appearance_score),
        }

    def rank_candidate(
        self,
        tight_embedding,
        context_embedding,
        reference_bank,
        embedder,
        detector_confidence=0.0,
    ):
        scores = []

        for ref_id, ref_data in reference_bank.items():
            ref_score = self._score_reference(
                tight_embedding=tight_embedding,
                context_embedding=context_embedding,
                reference_templates=ref_data["templates"],
                embedder=embedder,
            )

            scores.append(
                {
                    "reference_id": ref_id,
                    "tight_best_score": ref_score["tight_best_score"],
                    "context_best_score": ref_score["context_best_score"],
                    "appearance_score": ref_score["appearance_score"],
                }
            )

        scores.sort(key=lambda x: x["appearance_score"], reverse=True)

        if len(scores) == 0:
            return {
                "scores": [],
                "best_reference_id": None,
                "best_score": 0.0,
                "best_tight_score": 0.0,
                "best_context_score": 0.0,
                "second_reference_id": None,
                "second_score": 0.0,
                "margin": 0.0,
                "detector_confidence": float(detector_confidence),
            }

        best = scores[0]
        second = scores[1] if len(scores) > 1 else {
            "reference_id": None,
            "appearance_score": 0.0,
        }

        margin = float(best["appearance_score"] - second["appearance_score"])

        return {
            "scores": scores,
            "best_reference_id": best["reference_id"],
            "best_score": float(best["appearance_score"]),
            "best_tight_score": float(best["tight_best_score"]),
            "best_context_score": float(best["context_best_score"]),
            "second_reference_id": second["reference_id"],
            "second_score": float(second["appearance_score"]),
            "margin": margin,
            "detector_confidence": float(detector_confidence),
        }

    def decide(self, best_score: float, margin: float, detector_confidence: float) -> str:
        if detector_confidence < self.detector_conf_floor:
            return "reject_low_detector_conf"

        if best_score < self.min_similarity:
            return "reject_low_similarity"

        if (
            best_score >= self.strong_accept_similarity
            and margin >= self.strong_accept_margin
        ):
            return "accept_strong"

        if margin < self.min_margin:
            return "ambiguous_low_margin"

        if (
            self.ambiguity_score_lower <= best_score <= self.ambiguity_score_upper
            and margin <= self.ambiguity_margin_upper
        ):
            return "ambiguous_mid_similarity"

        return "accept"
