
import pickle
from pathlib import Path
from collections import defaultdict

from ..core.utils import (
    load_image_bgr,
    list_files_by_extensions,
    crop_xyxy,
    save_json,
    expand_bbox_xyxy,
    compute_iou,
    adjust_brightness_contrast,
    make_padded_template,
)
from ..detection.detector import YOLODetector
from ..embedding.embedder import DINOEmbedder
from ..embedding.batch_embedder import BatchEmbedder
from ..matching.topk_selector import TopKSelector
from ..geometry.geometric import GeometricVerifier
from ..tracking.track_manager import TrackManager
from ..proposals.grid_proposals import GridProposalGenerator
from ..proposals.proposal_fusion import ProposalFusion
from ..scoring.score_fusion import ScoreFusion


class InferenceEngine:
    def __init__(self, model_path, reference_dir, device, config, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.config = config
        self.device = device

        self.detector = YOLODetector(
            model_path=model_path,
            device=device,
            conf=config.detector["conf_threshold"],
            iou=config.detector["iou_threshold"],
            max_det=config.detector["max_det"],
            top_k_candidates=config.detector["top_k_candidates"],
            min_bbox_area=config.detector["min_bbox_area"],
            max_bbox_area_ratio=config.detector["max_bbox_area_ratio"],
            min_aspect_ratio=config.detector["min_aspect_ratio"],
            max_aspect_ratio=config.detector["max_aspect_ratio"],
        )

        self.embedder = DINOEmbedder(
            model_name=config.embedding["model_name"],
            device=device,
            input_size=config.embedding["input_size"],
            use_fp16=config.embedding["use_fp16"],
            normalize_embeddings=config.embedding["normalize_embeddings"],
            pooling=config.embedding["pooling"],
        )
        self.batch_embedder = BatchEmbedder(self.embedder)

        self.topk_selector = TopKSelector(
            top_k_per_reference=config.matching.get("top_k_per_reference_after_similarity", 15),
            top_k_for_geometry=config.matching.get("top_k_for_geometry", 3),
            min_appearance_for_geometry=config.geometry.get("min_appearance_for_geometry", 0.18),
        )

        self.geometry = GeometricVerifier(
            device=device,
            enabled=config.geometry["enabled"],
            pretrained=config.geometry["pretrained"],
            run_on_ambiguous_only=False,
            min_matches=config.geometry["min_matches"],
            min_inlier_ratio=config.geometry["min_inlier_ratio"],
            max_reproj_error=config.geometry["max_reproj_error"],
            max_image_size=config.geometry["max_image_size"],
        )

        self.tracker = TrackManager(config)

        self.grid_generator = GridProposalGenerator(
            enabled=config.grid_proposal.get("enabled", True),
            window_sizes=config.grid_proposal.get("window_sizes", [48, 64, 96, 128, 192, 256, 384]),
            stride_ratio=config.grid_proposal.get("stride_ratio", 0.50),
            aspect_ratios=config.grid_proposal.get("aspect_ratios", [0.75, 1.0, 1.33, 1.78]),
            max_grid_candidates=config.grid_proposal.get("max_grid_candidates", 120),
            balanced_sampling=config.grid_proposal.get("balanced_sampling", True),
            spatial_bins_x=config.grid_proposal.get("spatial_bins_x", 4),
            spatial_bins_y=config.grid_proposal.get("spatial_bins_y", 4),
        )

        self.proposal_fusion = ProposalFusion(
            max_total_candidates_per_frame=config.proposal.get("max_total_candidates_per_frame", 160),
            nms_iou_threshold=config.proposal.get("nms_iou_threshold", 0.65),
            min_bbox_area=config.proposal.get("min_bbox_area", 32),
            max_bbox_area_ratio=config.proposal.get("max_bbox_area_ratio", 0.70),
            min_aspect_ratio=config.proposal.get("min_aspect_ratio", 0.12),
            max_aspect_ratio=config.proposal.get("max_aspect_ratio", 7.0),
        )

        self.score_fusion = ScoreFusion(
            appearance_weight=config.fusion.get("appearance_score_weight", 0.80),
            geometry_weight=config.fusion.get("geometry_score_weight", 0.15),
            temporal_weight=config.fusion.get("temporal_score_weight", 0.05),
            mellin_weight=config.fusion.get("mellin_score_weight", 0.00),
            min_final_score=config.fusion.get("min_final_score", 0.30),
            min_final_margin=config.fusion.get("min_final_margin", 0.00),
            min_final_score_with_geometry=config.fusion.get("min_final_score_with_geometry", 0.24),
            strong_visual_accept_score=config.fusion.get("strong_visual_accept_score", 0.38),
            strong_visual_min_margin=config.fusion.get("strong_visual_min_margin", 0.00),
            yolo_source_bonus=config.fusion.get("yolo_source_bonus", 0.02),
            grid_source_bonus=config.fusion.get("grid_source_bonus", 0.00),
            local_source_penalty=config.fusion.get("local_source_penalty", 0.04),
        )

        self.context_expand_ratio = float(config.matching["context_expand_ratio"])

        self.reference_dir = Path(reference_dir)
        self.cache_dir = self.base_dir / config.paths["cache_dir"]
        self.reference_bank = self._load_or_build_reference_bank()

        self.tracker.initialize_references(sorted(list(self.reference_bank.keys())))

    def _get_reference_cache_path(self):
        return self.cache_dir / "reference_bank_v5_0_1.pkl"

    def _get_reference_summary_path(self):
        return self.cache_dir / "reference_bank_v5_0_1_summary.json"

    def _generate_template_images(self, image_bgr):
        templates = []
        templates.append(("orig", image_bgr))

        pad_ratio = float(self.config.embedding["template_pad_ratio"])
        templates.append(("pad", make_padded_template(image_bgr, pad_ratio=pad_ratio)))

        for bf in self.config.embedding["template_brightness_factors"]:
            if abs(float(bf) - 1.0) > 1e-6:
                templates.append(
                    (
                        f"brightness_{bf}",
                        adjust_brightness_contrast(
                            image_bgr,
                            brightness_factor=float(bf),
                            contrast_factor=1.0,
                        ),
                    )
                )

        for cf in self.config.embedding["template_contrast_factors"]:
            if abs(float(cf) - 1.0) > 1e-6:
                templates.append(
                    (
                        f"contrast_{cf}",
                        adjust_brightness_contrast(
                            image_bgr,
                            brightness_factor=1.0,
                            contrast_factor=float(cf),
                        ),
                    )
                )

        return templates

    def _load_or_build_reference_bank(self):
        reference_extensions = self.config.embedding["reference_extensions"]
        ref_paths = list_files_by_extensions(self.reference_dir, reference_extensions)

        if len(ref_paths) == 0:
            raise RuntimeError(f"Reference görüntüsü bulunamadı: {self.reference_dir}")

        cache_enabled = bool(self.config.embedding["cache_reference_embeddings"])
        cache_path = self._get_reference_cache_path()

        if cache_enabled and cache_path.exists():
            with cache_path.open("rb") as f:
                cache_data = pickle.load(f)

            cached_model_name = cache_data.get("model_name")
            cached_pooling = cache_data.get("pooling")
            cached_ref_ids = set(item["reference_id"] for item in cache_data.get("items", []))
            current_ref_ids = set(path.stem for path in ref_paths)

            if (
                cached_model_name == self.config.embedding["model_name"]
                and cached_pooling == self.config.embedding["pooling"]
                and cached_ref_ids == current_ref_ids
            ):
                bank = {}
                for item in cache_data["items"]:
                    ref_id = item["reference_id"]
                    path = Path(item["path"])
                    image_bgr = load_image_bgr(path)

                    bank[ref_id] = {
                        "reference_id": ref_id,
                        "path": str(path),
                        "image_bgr": image_bgr,
                        "templates": item["templates"],
                    }

                save_json(
                    {
                        "reference_count": len(bank),
                        "reference_ids": sorted(list(bank.keys())),
                        "cache_used": True,
                    },
                    self._get_reference_summary_path(),
                )
                return bank

        bank = {}
        cache_items = []

        for path in ref_paths:
            ref_id = path.stem
            image_bgr = load_image_bgr(path)
            template_images = self._generate_template_images(image_bgr)

            template_records = []
            for template_name, template_img in template_images:
                template_embedding = self.embedder.embed(template_img)
                template_records.append(
                    {
                        "name": template_name,
                        "embedding": template_embedding,
                    }
                )

            bank[ref_id] = {
                "reference_id": ref_id,
                "path": str(path),
                "image_bgr": image_bgr,
                "templates": template_records,
            }

            cache_items.append(
                {
                    "reference_id": ref_id,
                    "path": str(path),
                    "templates": template_records,
                }
            )

        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(
                    {
                        "model_name": self.config.embedding["model_name"],
                        "pooling": self.config.embedding["pooling"],
                        "items": cache_items,
                    },
                    f,
                )

        save_json(
            {
                "reference_count": len(bank),
                "reference_ids": sorted(list(bank.keys())),
                "cache_used": False,
            },
            self._get_reference_summary_path(),
        )

        return bank

    def _generate_yolo_proposals(self, frame_bgr, frame_name, frame_index):
        detections = self.detector.detect(frame_bgr)
        proposals = []

        for det in detections:
            rank = int(det["candidate_rank"])
            proposals.append(
                {
                    "candidate_id": f"{frame_name}__yolo__{rank}",
                    "frame_id": int(frame_index),
                    "frame_name": frame_name,
                    "candidate_rank": rank,
                    "source_type": "yolo",
                    "target_reference_id": None,
                    "bbox": [int(v) for v in det["bbox_xyxy"]],
                    "proposal_score": float(det["conf"]),
                    "detector_conf": float(det["conf"]),
                }
            )

        return proposals

    def _generate_grid_proposals(self, frame_bgr, frame_name, frame_index):
        image_h, image_w = frame_bgr.shape[:2]
        return self.grid_generator.generate(
            image_w=image_w,
            image_h=image_h,
            frame_name=frame_name,
            frame_index=frame_index,
        )

    def _generate_local_proposals(self, frame_bgr, frame_name, frame_index):
        if not self.tracker.local_search_enabled:
            return []

        image_h, image_w = frame_bgr.shape[:2]
        proposals = []

        for track in self.tracker.get_searchable_tracks():
            ref_id = track.reference_id
            prev_bbox = track.bbox

            if prev_bbox is None:
                continue

            boxes = self.tracker.temporal.generate_local_proposals(
                prev_bbox=prev_bbox,
                image_w=image_w,
                image_h=image_h,
                search_radius_factor=self.tracker.local_search_radius_factor,
                scale_factors=self.tracker.local_search_scale_factors,
                offsets=self.tracker.local_search_offsets,
            )

            boxes = sorted(
                boxes,
                key=lambda box: self.tracker.temporal.temporal_score(prev_bbox, box),
                reverse=True,
            )

            for rank, bbox in enumerate(boxes):
                temporal_score = self.tracker.temporal.temporal_score(prev_bbox, bbox)

                proposals.append(
                    {
                        "candidate_id": f"{frame_name}__local__{ref_id}__{rank}",
                        "frame_id": int(frame_index),
                        "frame_name": frame_name,
                        "candidate_rank": int(rank),
                        "source_type": "local_search",
                        "target_reference_id": ref_id,
                        "bbox": [int(v) for v in bbox],
                        "proposal_score": float(temporal_score),
                        "detector_conf": 0.0,
                        "temporal_score": float(temporal_score),
                    }
                )

        return proposals

    def _generate_all_proposals(self, frame_bgr, frame_name, frame_index):
        enabled_sources = set(self.config.proposal.get("enabled_sources", ["yolo", "grid", "local"]))

        proposals = []

        if "yolo" in enabled_sources:
            proposals.extend(self._generate_yolo_proposals(frame_bgr, frame_name, frame_index))

        if "grid" in enabled_sources:
            proposals.extend(self._generate_grid_proposals(frame_bgr, frame_name, frame_index))

        if "local" in enabled_sources:
            proposals.extend(self._generate_local_proposals(frame_bgr, frame_name, frame_index))

        image_h, image_w = frame_bgr.shape[:2]
        kept, proposal_logs = self.proposal_fusion.fuse(
            proposals=proposals,
            image_w=image_w,
            image_h=image_h,
        )

        return kept, proposal_logs

    def _score_reference_templates(self, tight_embedding, context_embedding, reference_templates):
        tight_scores = []
        context_scores = []

        for template in reference_templates:
            ref_emb = template["embedding"]
            tight_scores.append(self.embedder.cosine(tight_embedding, ref_emb))
            context_scores.append(self.embedder.cosine(context_embedding, ref_emb))

        tight_best = max(tight_scores) if len(tight_scores) else 0.0
        context_best = max(context_scores) if len(context_scores) else 0.0

        appearance_score = (
            float(self.config.matching["tight_weight"]) * float(tight_best)
            + float(self.config.matching["context_weight"]) * float(context_best)
        )

        return float(tight_best), float(context_best), float(appearance_score)

    def _score_all_candidates(self, frame_bgr, proposals, tracks_before):
        image_h, image_w = frame_bgr.shape[:2]

        candidate_logs = []
        similarity_logs = []
        scored_pairs = []
        crop_cache = {}

        tight_crops = []
        context_crops = []
        valid_props = []

        for proposal in proposals:
            bbox = proposal["bbox"]
            context_bbox = expand_bbox_xyxy(
                bbox,
                image_w=image_w,
                image_h=image_h,
                expand_ratio=self.context_expand_ratio,
            )

            tight_crop = crop_xyxy(frame_bgr, bbox)
            context_crop = crop_xyxy(frame_bgr, context_bbox)

            record = {
                "frame_id": proposal["frame_id"],
                "frame_name": proposal["frame_name"],
                "candidate_id": proposal["candidate_id"],
                "candidate_rank": proposal["candidate_rank"],
                "source_type": proposal["source_type"],
                "target_reference_id": proposal.get("target_reference_id"),
                "track_state_before": "unknown",

                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],

                "context_bbox_x1": context_bbox[0],
                "context_bbox_y1": context_bbox[1],
                "context_bbox_x2": context_bbox[2],
                "context_bbox_y2": context_bbox[3],

                "detector_conf": float(proposal.get("detector_conf", 0.0)),
                "proposal_score": float(proposal.get("proposal_score", 0.0)),

                "best_reference_id": None,
                "best_score": 0.0,
                "best_tight_score": 0.0,
                "best_context_score": 0.0,
                "second_reference_id": None,
                "second_score": 0.0,
                "margin": 0.0,

                "temporal_iou": 0.0,
                "temporal_score": float(proposal.get("temporal_score", 0.0)),

                "geometry_used": False,
                "geometry_num_matches": 0,
                "geometry_num_inliers": 0,
                "geometry_inlier_ratio": 0.0,
                "geometry_pass": False,
                "geometry_score": 0.0,

                "mellin_score": 0.0,
                "final_score": 0.0,
                "compare_score": 0.0,
                "decision": "reject_invalid_crop",
                "final_reference_id": None,

                "selected_for_reference": False,
                "selected_for_output": False,
            }

            if tight_crop is None or context_crop is None:
                candidate_logs.append(record)
                continue

            valid_props.append((proposal, record, context_bbox))
            tight_crops.append(tight_crop)
            context_crops.append(context_crop)

            crop_cache[proposal["candidate_id"]] = {
                "tight_crop": tight_crop,
                "context_crop": context_crop,
                "context_bbox": context_bbox,
            }

        if len(valid_props) == 0:
            return candidate_logs, similarity_logs, scored_pairs, crop_cache

        tight_embeddings = self.batch_embedder.embed_many(tight_crops)
        context_embeddings = self.batch_embedder.embed_many(context_crops)

        for idx, (proposal, record, context_bbox) in enumerate(valid_props):
            tight_embedding = tight_embeddings[idx]
            context_embedding = context_embeddings[idx]

            ref_scores = []

            for ref_id, ref_data in self.reference_bank.items():
                tight_best, context_best, appearance_score = self._score_reference_templates(
                    tight_embedding=tight_embedding,
                    context_embedding=context_embedding,
                    reference_templates=ref_data["templates"],
                )

                ref_scores.append(
                    {
                        "reference_id": ref_id,
                        "tight_score": tight_best,
                        "context_score": context_best,
                        "appearance_score": appearance_score,
                    }
                )

            ref_scores = sorted(ref_scores, key=lambda x: x["appearance_score"], reverse=True)

            best = ref_scores[0]
            second = ref_scores[1] if len(ref_scores) > 1 else {
                "reference_id": None,
                "appearance_score": 0.0,
            }

            margin_for_best = float(best["appearance_score"] - second["appearance_score"])

            record["best_reference_id"] = best["reference_id"]
            record["best_score"] = float(best["appearance_score"])
            record["best_tight_score"] = float(best["tight_score"])
            record["best_context_score"] = float(best["context_score"])
            record["second_reference_id"] = second["reference_id"]
            record["second_score"] = float(second["appearance_score"])
            record["margin"] = margin_for_best
            record["decision"] = "scored"

            if best["reference_id"] in tracks_before:
                record["track_state_before"] = tracks_before[best["reference_id"]].get("state", "ABSENT")
                prev_bbox = tracks_before[best["reference_id"]].get("bbox", None)
                if prev_bbox is not None:
                    record["temporal_iou"] = float(compute_iou(prev_bbox, proposal["bbox"]))
                    record["temporal_score"] = float(
                        self.tracker.temporal.temporal_score(prev_bbox, proposal["bbox"])
                    )

            target_ref = proposal.get("target_reference_id")
            if proposal["source_type"] == "local_search" and target_ref is not None:
                if best["reference_id"] != target_ref:
                    record["decision"] = "reject_local_wrong_reference"

            candidate_logs.append(record)

            max_other_by_ref = {}
            for ref_score in ref_scores:
                ref_id = ref_score["reference_id"]
                others = [x["appearance_score"] for x in ref_scores if x["reference_id"] != ref_id]
                second_best = max(others) if len(others) else 0.0
                max_other_by_ref[ref_id] = second_best

            for ref_score in ref_scores:
                ref_id = ref_score["reference_id"]
                margin = float(ref_score["appearance_score"] - max_other_by_ref[ref_id])

                sim_row = {
                    "frame_id": proposal["frame_id"],
                    "frame_name": proposal["frame_name"],
                    "candidate_id": proposal["candidate_id"],
                    "source_type": proposal["source_type"],
                    "target_reference_id": proposal.get("target_reference_id"),
                    "reference_id": ref_id,
                    "bbox": proposal["bbox"],
                    "bbox_x1": proposal["bbox"][0],
                    "bbox_y1": proposal["bbox"][1],
                    "bbox_x2": proposal["bbox"][2],
                    "bbox_y2": proposal["bbox"][3],
                    "context_bbox": context_bbox,
                    "detector_conf": float(proposal.get("detector_conf", 0.0)),
                    "proposal_score": float(proposal.get("proposal_score", 0.0)),
                    "tight_score": float(ref_score["tight_score"]),
                    "context_score": float(ref_score["context_score"]),
                    "appearance_score": float(ref_score["appearance_score"]),
                    "visual_score": float(ref_score["appearance_score"]),
                    "margin": float(margin),
                    "temporal_score": float(record["temporal_score"]),
                    "geometry_score": 0.0,
                    "geometry_used": False,
                    "geometry_pass": False,
                    "mellin_score": 0.0,
                    "rank_for_reference": None,
                    "final_score": 0.0,
                    "compare_score": 0.0,
                    "decision": "similarity_scored",
                }

                if proposal["source_type"] == "local_search" and target_ref is not None and ref_id != target_ref:
                    sim_row["decision"] = "reject_local_wrong_reference"

                similarity_logs.append(dict(sim_row))
                scored_pairs.append(sim_row)

        return candidate_logs, similarity_logs, scored_pairs, crop_cache

    def _apply_rank_to_similarity_logs(self, similarity_logs, scored_pairs):
        rank_map = {}
        for row in scored_pairs:
            rank_map[(row["candidate_id"], row["reference_id"])] = row.get("rank_for_reference")

        for row in similarity_logs:
            row["rank_for_reference"] = rank_map.get((row["candidate_id"], row["reference_id"]))

        return similarity_logs

    def _run_geometry(self, geometry_by_ref, crop_cache):
        geometry_logs = []
        geometry_map = {}

        if not self.config.geometry.get("enabled", True):
            return geometry_map, geometry_logs

        if not self.geometry.is_available():
            return geometry_map, geometry_logs

        seen = set()

        for ref_id, rows in geometry_by_ref.items():
            for row in rows:
                key = (row["candidate_id"], ref_id)
                if key in seen:
                    continue
                seen.add(key)

                crop_info = crop_cache.get(row["candidate_id"])
                if crop_info is None:
                    continue

                context_crop = crop_info["context_crop"]

                geom_result = self.geometry.verify(
                    self.reference_bank[ref_id]["image_bgr"],
                    context_crop,
                )

                geom_score = float(geom_result.get("geom_score", 0.0))
                geom_pass = bool(geom_result.get("geom_pass", False))

                geometry_map[key] = {
                    "geometry_used": bool(geom_result.get("used", True)),
                    "geometry_num_matches": int(geom_result.get("num_matches", 0)),
                    "geometry_num_inliers": int(geom_result.get("num_inliers", 0)),
                    "geometry_inlier_ratio": float(geom_result.get("inlier_ratio", 0.0)),
                    "geometry_pass": geom_pass,
                    "geometry_score": geom_score,
                }

                geometry_logs.append(
                    {
                        "frame_id": row["frame_id"],
                        "frame_name": row["frame_name"],
                        "candidate_id": row["candidate_id"],
                        "reference_id": ref_id,
                        "source_type": row["source_type"],
                        "rank_for_reference": row.get("rank_for_reference"),
                        "appearance_score": float(row.get("appearance_score", 0.0)),
                        "geometry_used": bool(geom_result.get("used", True)),
                        "geometry_num_matches": int(geom_result.get("num_matches", 0)),
                        "geometry_num_inliers": int(geom_result.get("num_inliers", 0)),
                        "geometry_inlier_ratio": float(geom_result.get("inlier_ratio", 0.0)),
                        "geometry_pass": geom_pass,
                        "geometry_score": geom_score,
                    }
                )

        return geometry_map, geometry_logs

    def _is_accept_decision(self, decision):
        return str(decision).startswith("accept")

    def _fuse_and_decide(self, topk_by_ref, geometry_map):
        fusion_logs = []
        accepted_by_reference = defaultdict(list)

        for ref_id, rows in topk_by_ref.items():
            for row in rows:
                key = (row["candidate_id"], ref_id)

                geom = geometry_map.get(key, None)
                if geom is not None:
                    row.update(geom)

                final_score, compare_score = self.score_fusion.fuse(row)

                row["final_score"] = float(final_score)
                row["compare_score"] = float(compare_score)

                decision = self.score_fusion.decide(row)

                if row.get("source_type") == "local_search":
                    target_ref = row.get("target_reference_id")
                    if target_ref is not None and target_ref != ref_id:
                        decision = "reject_local_wrong_reference"
                    elif row["appearance_score"] < self.tracker.local_min_similarity:
                        decision = "reject_local_low_similarity"
                    elif row["margin"] < self.tracker.local_min_margin:
                        decision = "reject_local_low_margin"
                    elif row["temporal_score"] < self.tracker.local_min_temporal_score:
                        decision = "reject_local_low_temporal"
                    elif final_score < self.tracker.local_min_final_score:
                        decision = "reject_local_low_final_score"

                row["decision"] = decision

                fusion_log = {
                    "frame_id": row["frame_id"],
                    "frame_name": row["frame_name"],
                    "candidate_id": row["candidate_id"],
                    "source_type": row["source_type"],
                    "reference_id": ref_id,
                    "rank_for_reference": row.get("rank_for_reference"),
                    "appearance_score": float(row.get("appearance_score", 0.0)),
                    "geometry_score": float(row.get("geometry_score", 0.0)),
                    "geometry_pass": bool(row.get("geometry_pass", False)),
                    "mellin_score": float(row.get("mellin_score", 0.0)),
                    "temporal_score": float(row.get("temporal_score", 0.0)),
                    "margin": float(row.get("margin", 0.0)),
                    "final_score": float(final_score),
                    "compare_score": float(compare_score),
                    "decision": decision,
                }
                fusion_logs.append(fusion_log)

                if self._is_accept_decision(decision):
                    accepted_by_reference[ref_id].append(dict(row))

        return accepted_by_reference, fusion_logs

    def _pick_best_candidate(self, candidates):
        if len(candidates) == 0:
            return None
        return max(candidates, key=lambda x: float(x.get("compare_score", 0.0)))

    def _candidate_ok_for_track(self, track, cand):
        if cand is None:
            return False

        decision = cand.get("decision", "")
        source = cand.get("source_type")

        if not self._is_accept_decision(decision):
            return False

        if source == "local_search":
            return True

        if track.state == "ABSENT":
            return (
                float(cand.get("final_score", 0.0)) >= self.tracker.birth_min_score
                and float(cand.get("margin", 0.0)) >= self.tracker.birth_min_margin
            )

        if track.state == "TRACKING":
            return (
                float(cand.get("appearance_score", 0.0)) >= self.tracker.tracked_min_detector_score
                and float(cand.get("margin", 0.0)) >= self.tracker.tracked_min_detector_margin
            )

        if track.state == "LOST":
            return (
                float(cand.get("appearance_score", 0.0)) >= self.tracker.lost_min_detector_score
                and float(cand.get("margin", 0.0)) >= self.tracker.lost_min_detector_margin
            )

        return False

    def _select_candidate_for_reference(self, track, candidates):
        global_candidates = [c for c in candidates if c.get("source_type") != "local_search"]
        local_candidates = [c for c in candidates if c.get("source_type") == "local_search"]

        global_best = self._pick_best_candidate(global_candidates)
        local_best = self._pick_best_candidate(local_candidates)

        global_ok = self._candidate_ok_for_track(track, global_best)
        local_ok = self._candidate_ok_for_track(track, local_best)

        if track.state == "ABSENT":
            return global_best if global_ok else None

        need_global_refresh = (
            track.frames_since_detector >= self.tracker.force_detector_refresh_every_n_frames
            or track.local_only_streak >= self.tracker.max_local_only_streak
        )

        if need_global_refresh:
            return global_best if global_ok else None

        if global_ok and local_ok:
            if global_best["compare_score"] >= local_best["compare_score"] - self.tracker.detector_preference_delta:
                return global_best
            return local_best

        if global_ok:
            return global_best

        if local_ok:
            return local_best

        return None

    def _resolve_cross_reference_conflicts(self, selected_candidates):
        kept = []

        for cand in sorted(selected_candidates, key=lambda x: float(x.get("compare_score", 0.0)), reverse=True):
            has_conflict = False
            for kept_cand in kept:
                iou_val = compute_iou(cand["bbox"], kept_cand["bbox"])
                if iou_val >= self.tracker.cross_reference_iou_threshold:
                    has_conflict = True
                    break
            if not has_conflict:
                kept.append(cand)

        return kept

    def process_frame(self, frame_bgr, frame_name: str, frame_index: int):
        tracks_before = self.tracker.snapshot_states()

        proposals, proposal_logs = self._generate_all_proposals(
            frame_bgr=frame_bgr,
            frame_name=frame_name,
            frame_index=frame_index,
        )

        candidate_logs, similarity_logs, scored_pairs, crop_cache = self._score_all_candidates(
            frame_bgr=frame_bgr,
            proposals=proposals,
            tracks_before=tracks_before,
        )

        topk_by_ref, geometry_by_ref = self.topk_selector.select(scored_pairs)
        similarity_logs = self._apply_rank_to_similarity_logs(similarity_logs, scored_pairs)

        geometry_map, geometry_logs = self._run_geometry(
            geometry_by_ref=geometry_by_ref,
            crop_cache=crop_cache,
        )

        accepted_by_reference, fusion_logs = self._fuse_and_decide(
            topk_by_ref=topk_by_ref,
            geometry_map=geometry_map,
        )

        selected_for_reference = []

        for ref_id in sorted(self.reference_bank.keys()):
            track = self.tracker.get_track(ref_id)
            candidates = accepted_by_reference.get(ref_id, [])

            chosen = self._select_candidate_for_reference(
                track=track,
                candidates=candidates,
            )

            if chosen is not None:
                selected_for_reference.append(chosen)

        selected_output = self._resolve_cross_reference_conflicts(selected_for_reference)

        matched_reference_ids = set()
        output_objects = []

        for cand in selected_output:
            ref_id = cand["reference_id"]
            matched_reference_ids.add(ref_id)

            self.tracker.update_with_candidate(
                reference_id=ref_id,
                bbox=cand["bbox"],
                score=cand["final_score"],
                frame_index=frame_index,
                source=cand["source_type"],
                appearance_score=cand["appearance_score"],
                detector_conf=cand.get("detector_conf", 0.0),
                temporal_score=cand.get("temporal_score", 0.0),
            )

            track_after = self.tracker.get_track(ref_id)

            output_objects.append(
                {
                    "reference_id": ref_id,
                    "track_id": track_after.track_id,
                    "bbox": [int(v) for v in cand["bbox"]],
                    "score": float(cand["final_score"]),
                    "appearance_score": float(cand["appearance_score"]),
                    "detector_conf": float(cand.get("detector_conf", 0.0)),
                    "temporal_score": float(cand.get("temporal_score", 0.0)),
                    "source_type": cand["source_type"],
                    "decision": cand.get("decision"),
                }
            )

        self.tracker.mark_unmatched_tracks(
            matched_reference_ids=matched_reference_ids,
            frame_index=frame_index,
        )

        tracks_after = self.tracker.snapshot_states()

        frame_summary = {
            "frame_id": frame_index,
            "frame_name": frame_name,
            "num_raw_proposals": len(proposal_logs),
            "num_kept_proposals": len(proposals),
            "num_candidate_logs": len(candidate_logs),
            "num_similarity_pairs": len(similarity_logs),
            "num_fusion_rows": len(fusion_logs),
            "num_accepted": len(output_objects),
            "num_absent_predictions": 1 if len(output_objects) == 0 else 0,
            "num_geometry_attempted": len(geometry_logs),
            "num_geometry_passed": sum(1 for g in geometry_logs if g.get("geometry_pass")),
            "num_tracking_active": sum(1 for t in tracks_after.values() if t["state"] == "TRACKING"),
            "num_tracking_lost": sum(1 for t in tracks_after.values() if t["state"] == "LOST"),
        }

        frame_result = {
            "frame_name": frame_name,
            "objects": output_objects,
        }

        frame_debug = {
            "frame_summary": frame_summary,
            "proposal_logs": proposal_logs,
            "candidate_logs": candidate_logs,
            "similarity_logs": similarity_logs,
            "geometry_logs": geometry_logs,
            "fusion_logs": fusion_logs,
            "final_objects": output_objects,
            "track_snapshot_before": tracks_before,
            "track_snapshot_after": tracks_after,
        }

        return {
            "frame_result": frame_result,
            "frame_summary": frame_summary,
            "proposal_logs": proposal_logs,
            "candidate_logs": candidate_logs,
            "similarity_logs": similarity_logs,
            "geometry_logs": geometry_logs,
            "fusion_logs": fusion_logs,
            "frame_debug": frame_debug,
        }
