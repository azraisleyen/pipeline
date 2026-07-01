
import time
from pathlib import Path
import cv2

from .frame_loader import FrameLoader
from .inference_engine import InferenceEngine
from ..core.utils import (
    save_json,
    save_csv_rows,
    draw_objects_on_image,
    ensure_dir,
)
from ..output.official_adapter import save_official_json


class VideoRunner:
    def __init__(
        self,
        video_path,
        frames_dir,
        model_path,
        reference_dir,
        output_internal_json,
        output_official_json,
        frame_log_csv,
        proposal_log_csv,
        similarity_log_csv,
        candidate_log_csv,
        fusion_log_csv,
        geometry_log_csv,
        track_log_csv,
        visualization_dir,
        debug_frame_dir,
        device,
        config,
        base_dir,
        logger=None,
    ):
        self.base_dir = Path(base_dir)
        self.logger = logger
        self.config = config

        self.loader = FrameLoader(video_path, frames_dir)

        self.engine = InferenceEngine(
            model_path=model_path,
            reference_dir=reference_dir,
            device=device,
            config=config,
            base_dir=self.base_dir,
        )

        self.output_internal_json = Path(output_internal_json)
        self.output_official_json = Path(output_official_json)

        self.frame_log_csv = Path(frame_log_csv)
        self.proposal_log_csv = Path(proposal_log_csv)
        self.similarity_log_csv = Path(similarity_log_csv)
        self.candidate_log_csv = Path(candidate_log_csv)
        self.fusion_log_csv = Path(fusion_log_csv)
        self.geometry_log_csv = Path(geometry_log_csv)
        self.track_log_csv = Path(track_log_csv)

        self.visualization_dir = Path(visualization_dir)
        self.debug_frame_dir = Path(debug_frame_dir)

        for p in [
            self.output_internal_json.parent,
            self.output_official_json.parent,
            self.frame_log_csv.parent,
            self.proposal_log_csv.parent,
            self.similarity_log_csv.parent,
            self.candidate_log_csv.parent,
            self.fusion_log_csv.parent,
            self.geometry_log_csv.parent,
            self.track_log_csv.parent,
            self.visualization_dir,
            self.debug_frame_dir,
        ]:
            ensure_dir(p)

    def _log(self, message, *args):
        if self.logger is not None:
            self.logger.info(message, *args)
        else:
            print(message % args if args else message)

    def _save_visualization(self, frame_bgr, frame_name, objects):
        if not self.config.runtime.get("save_visualizations", True):
            return

        vis = draw_objects_on_image(frame_bgr, objects)
        out_path = self.visualization_dir / frame_name
        cv2.imwrite(str(out_path), vis)

    def run(self):
        frame_paths = self.loader.get_frame_list()

        results = {"frames": []}

        frame_rows = []
        proposal_rows = []
        similarity_rows = []
        candidate_rows = []
        fusion_rows = []
        geometry_rows = []
        track_rows = []

        total_time_ms = 0.0

        self._log("Toplam frame sayısı: %d", len(frame_paths))

        for frame_id, frame_path in enumerate(frame_paths):
            frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            frame_name = frame_path.name

            if frame_bgr is None:
                frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "frame_name": frame_name,
                        "num_raw_proposals": 0,
                        "num_kept_proposals": 0,
                        "num_candidate_logs": 0,
                        "num_similarity_pairs": 0,
                        "num_fusion_rows": 0,
                        "num_accepted": 0,
                        "num_absent_predictions": 1,
                        "num_geometry_attempted": 0,
                        "num_geometry_passed": 0,
                        "num_tracking_active": 0,
                        "num_tracking_lost": 0,
                        "frame_time_ms": 0.0,
                        "fps": 0.0,
                        "status": "read_error",
                    }
                )
                continue

            start_time = time.time()

            try:
                output = self.engine.process_frame(
                    frame_bgr=frame_bgr,
                    frame_name=frame_name,
                    frame_index=frame_id,
                )
            except Exception as exc:
                if self.config.runtime.get("fail_fast", False):
                    raise

                frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "frame_name": frame_name,
                        "num_raw_proposals": 0,
                        "num_kept_proposals": 0,
                        "num_candidate_logs": 0,
                        "num_similarity_pairs": 0,
                        "num_fusion_rows": 0,
                        "num_accepted": 0,
                        "num_absent_predictions": 1,
                        "num_geometry_attempted": 0,
                        "num_geometry_passed": 0,
                        "num_tracking_active": 0,
                        "num_tracking_lost": 0,
                        "frame_time_ms": 0.0,
                        "fps": 0.0,
                        "status": f"process_error: {exc}",
                    }
                )
                self._log("Frame hatası: %s | %s", frame_name, str(exc))
                continue

            elapsed_ms = (time.time() - start_time) * 1000.0
            total_time_ms += elapsed_ms
            fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

            frame_result = output["frame_result"]
            frame_summary = output["frame_summary"]
            frame_debug = output["frame_debug"]

            results["frames"].append(frame_result)

            proposal_rows.extend(output["proposal_logs"])
            similarity_rows.extend(output["similarity_logs"])
            candidate_rows.extend(output["candidate_logs"])
            fusion_rows.extend(output["fusion_logs"])
            geometry_rows.extend(output["geometry_logs"])

            frame_rows.append(
                {
                    "frame_id": frame_id,
                    "frame_name": frame_name,
                    "num_raw_proposals": frame_summary["num_raw_proposals"],
                    "num_kept_proposals": frame_summary["num_kept_proposals"],
                    "num_candidate_logs": frame_summary["num_candidate_logs"],
                    "num_similarity_pairs": frame_summary["num_similarity_pairs"],
                    "num_fusion_rows": frame_summary["num_fusion_rows"],
                    "num_accepted": frame_summary["num_accepted"],
                    "num_absent_predictions": frame_summary["num_absent_predictions"],
                    "num_geometry_attempted": frame_summary["num_geometry_attempted"],
                    "num_geometry_passed": frame_summary["num_geometry_passed"],
                    "num_tracking_active": frame_summary["num_tracking_active"],
                    "num_tracking_lost": frame_summary["num_tracking_lost"],
                    "frame_time_ms": round(elapsed_ms, 4),
                    "fps": round(fps, 4),
                    "status": "ok",
                }
            )

            track_snapshot_after = frame_debug["track_snapshot_after"]
            for ref_id, track_info in track_snapshot_after.items():
                bbox = track_info.get("bbox", None)
                raw_bbox = track_info.get("raw_bbox", None)

                track_rows.append(
                    {
                        "frame_id": frame_id,
                        "frame_name": frame_name,
                        "reference_id": ref_id,
                        "track_id": track_info.get("track_id", -1),
                        "state": track_info.get("state", "UNKNOWN"),
                        "bbox_x1": bbox[0] if bbox is not None else None,
                        "bbox_y1": bbox[1] if bbox is not None else None,
                        "bbox_x2": bbox[2] if bbox is not None else None,
                        "bbox_y2": bbox[3] if bbox is not None else None,
                        "raw_bbox_x1": raw_bbox[0] if raw_bbox is not None else None,
                        "raw_bbox_y1": raw_bbox[1] if raw_bbox is not None else None,
                        "raw_bbox_x2": raw_bbox[2] if raw_bbox is not None else None,
                        "raw_bbox_y2": raw_bbox[3] if raw_bbox is not None else None,
                        "hits": track_info.get("hits", 0),
                        "lost_count": track_info.get("lost_count", 0),
                        "age": track_info.get("age", 0),
                        "local_only_streak": track_info.get("local_only_streak", 0),
                        "frames_since_detector": track_info.get("frames_since_detector", 0),
                        "last_score": track_info.get("last_score", 0.0),
                        "last_appearance_score": track_info.get("last_appearance_score", 0.0),
                        "last_detector_conf": track_info.get("last_detector_conf", 0.0),
                        "last_temporal_score": track_info.get("last_temporal_score", 0.0),
                        "last_source": track_info.get("last_source", "none"),
                    }
                )

            if self.config.runtime.get("save_debug_outputs", True):
                debug_path = self.debug_frame_dir / f"{Path(frame_name).stem}.json"
                save_json(frame_debug, debug_path)

            self._save_visualization(
                frame_bgr=frame_bgr,
                frame_name=frame_name,
                objects=frame_result["objects"],
            )

            if frame_id % 50 == 0:
                self._log(
                    "[%d / %d] frame=%s | proposals=%d | accepted=%d | geometry=%d | active=%d | time_ms=%.2f",
                    frame_id + 1,
                    len(frame_paths),
                    frame_name,
                    frame_summary["num_kept_proposals"],
                    frame_summary["num_accepted"],
                    frame_summary["num_geometry_attempted"],
                    frame_summary["num_tracking_active"],
                    elapsed_ms,
                )

        save_json(results, self.output_internal_json)

        if self.config.output.get("write_official_json", True):
            save_official_json(
                internal_results=results,
                output_path=self.output_official_json,
                object_key=self.config.output.get("official_object_key", "detected_undefined_objects"),
            )

        save_csv_rows(frame_rows, self.frame_log_csv)
        save_csv_rows(proposal_rows, self.proposal_log_csv)
        save_csv_rows(similarity_rows, self.similarity_log_csv)
        save_csv_rows(candidate_rows, self.candidate_log_csv)
        save_csv_rows(fusion_rows, self.fusion_log_csv)
        save_csv_rows(geometry_rows, self.geometry_log_csv)
        save_csv_rows(track_rows, self.track_log_csv)

        avg_fps = (len(frame_paths) * 1000.0 / total_time_ms) if total_time_ms > 0 else 0.0

        summary = {
            "num_frames": len(frame_paths),
            "avg_fps": avg_fps,
            "total_time_ms": total_time_ms,
        }

        self._log("Video işleme tamamlandı.")
        self._log("Ortalama FPS: %.4f", avg_fps)

        return summary
