from __future__ import annotations

import math


class TranslationHealthPolicy:
    def __init__(self, use_upstream_when_valid=True, keep_last_valid=True):
        self.use_upstream = use_upstream_when_valid
        self.keep_last = keep_last_valid

    def valid(self, tr):
        return isinstance(tr, dict) and all(
            isinstance(tr.get(k), (int, float)) and math.isfinite(float(tr[k]))
            for k in ["translation_x", "translation_y", "translation_z"]
        )

    def choose(self, context, state, default):
        if self.use_upstream and context.health_status.get("translation_valid") and self.valid(context.upstream_translation):
            return {k: float(context.upstream_translation[k]) for k in default}
        if self.keep_last and state.last_valid and self.valid(state.last_valid):
            return {k: float(state.last_valid[k]) for k in default}
        return {k: float(default[k]) for k in default}
