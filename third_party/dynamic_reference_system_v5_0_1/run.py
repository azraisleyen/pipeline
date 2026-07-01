
from pathlib import Path
import json

from src.core.config import Config
from src.core.device import select_device, set_global_seed
from src.core.logger import setup_logger
from src.core.utils import ensure_dirs, copy_file, save_json
from src.pipeline.video_runner import VideoRunner
from src.evaluation.evaluator import Evaluator


def main():
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "configs" / "default.yaml"

    config = Config(config_path)

    set_global_seed(
        seed=int(config.runtime["seed"]),
        deterministic=bool(config.runtime["deterministic"]),
    )

    device = select_device(config.runtime["device"])

    video_path = config.resolve_path(base_dir, config.paths["video_path"])
    frames_dir = config.resolve_path(base_dir, config.paths["frames_dir"])
    reference_dir = config.resolve_path(base_dir, config.paths["reference_dir"])
    gt_path = config.resolve_path(base_dir, config.paths["gt_json"])

    model_path = config.resolve_path(base_dir, config.detector["model_path"])

    output_internal_json = config.resolve_path(base_dir, config.paths["output_internal_json"])
    output_official_json = config.resolve_path(base_dir, config.paths["output_official_json"])

    frame_log_csv = config.resolve_path(base_dir, config.paths["frame_log_csv"])
    proposal_log_csv = config.resolve_path(base_dir, config.paths["proposal_log_csv"])
    similarity_log_csv = config.resolve_path(base_dir, config.paths["similarity_log_csv"])
    candidate_log_csv = config.resolve_path(base_dir, config.paths["candidate_log_csv"])
    fusion_log_csv = config.resolve_path(base_dir, config.paths["fusion_log_csv"])
    geometry_log_csv = config.resolve_path(base_dir, config.paths["geometry_log_csv"])
    track_log_csv = config.resolve_path(base_dir, config.paths["track_log_csv"])

    evaluation_json = config.resolve_path(base_dir, config.paths["evaluation_json"])
    per_reference_csv = config.resolve_path(base_dir, config.paths["per_reference_csv"])
    fp_csv = config.resolve_path(base_dir, config.paths["fp_csv"])
    fn_csv = config.resolve_path(base_dir, config.paths["fn_csv"])

    visualization_dir = config.resolve_path(base_dir, config.paths["visualization_dir"])
    debug_frame_dir = config.resolve_path(base_dir, config.paths["debug_frame_dir"])
    cache_dir = config.resolve_path(base_dir, config.paths["cache_dir"])
    config_snapshot_dir = config.resolve_path(base_dir, config.paths["config_snapshot_dir"])

    ensure_dirs(
        [
            output_internal_json.parent,
            output_official_json.parent,
            frame_log_csv.parent,
            proposal_log_csv.parent,
            similarity_log_csv.parent,
            candidate_log_csv.parent,
            fusion_log_csv.parent,
            geometry_log_csv.parent,
            track_log_csv.parent,
            evaluation_json.parent,
            per_reference_csv.parent,
            fp_csv.parent,
            fn_csv.parent,
            visualization_dir,
            debug_frame_dir,
            cache_dir,
            config_snapshot_dir,
            base_dir / "logs",
        ]
    )

    logger = setup_logger(
        log_dir=base_dir / "logs",
        run_name=str(config.runtime.get("run_name", "v5_0_1_recall_balanced_grid")),
    )

    logger.info("====================================")
    logger.info("Dynamic Reference System V5.0.1 Starting")
    logger.info("Base dir: %s", base_dir)
    logger.info("Device: %s", device)
    logger.info("Reference dir: %s", reference_dir)
    logger.info("Frames dir: %s", frames_dir)
    logger.info("Model path: %s", model_path)
    logger.info("====================================")

    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference klasörü bulunamadı: {reference_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")

    if (not video_path.exists()) and (not frames_dir.exists()):
        raise FileNotFoundError(
            f"Ne video ne de frames klasörü bulunabildi. Video: {video_path} | Frames: {frames_dir}"
        )

    config_snapshot_path = config_snapshot_dir / "default_used_v5_0_1.yaml"
    copy_file(config_path, config_snapshot_path)
    logger.info("Config snapshot kaydedildi: %s", config_snapshot_path)

    runner = VideoRunner(
        video_path=video_path,
        frames_dir=frames_dir,
        model_path=model_path,
        reference_dir=reference_dir,
        output_internal_json=output_internal_json,
        output_official_json=output_official_json,
        frame_log_csv=frame_log_csv,
        proposal_log_csv=proposal_log_csv,
        similarity_log_csv=similarity_log_csv,
        candidate_log_csv=candidate_log_csv,
        fusion_log_csv=fusion_log_csv,
        geometry_log_csv=geometry_log_csv,
        track_log_csv=track_log_csv,
        visualization_dir=visualization_dir,
        debug_frame_dir=debug_frame_dir,
        device=device,
        config=config,
        base_dir=base_dir,
        logger=logger,
    )

    run_summary = runner.run()
    logger.info("Frame işleme tamamlandı. Özet: %s", json.dumps(run_summary, indent=2))

    if gt_path.exists():
        evaluator = Evaluator()
        metrics = evaluator.evaluate(
            gt_path=gt_path,
            pred_path=output_internal_json,
            threshold=float(config.evaluation["iou_threshold"]),
            per_reference_csv_path=per_reference_csv,
            fp_csv_path=fp_csv,
            fn_csv_path=fn_csv,
        )
        save_json(metrics, evaluation_json)

        logger.info("Evaluation tamamlandı.")
        logger.info("Precision: %.6f", metrics["precision"])
        logger.info("Recall: %.6f", metrics["recall"])
        logger.info("F1: %.6f", metrics["f1"])
        logger.info("mAP@0.5: %.6f", metrics["map50"])
        logger.info("ID Switch: %d", metrics.get("id_switch", 0))
    else:
        logger.warning("GT bulunamadı, evaluation atlandı: %s", gt_path)

    logger.info("====================================")
    logger.info("Dynamic Reference System V5.0.1 Finished")
    logger.info("====================================")


if __name__ == "__main__":
    main()
