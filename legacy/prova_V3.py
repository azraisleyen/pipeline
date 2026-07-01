from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path(__file__).resolve().parent
MODEL_DIR = SIM_ROOT / "last_models"
RESULTS_DIR = SIM_ROOT / "results_V3"

DEFAULT_VIDEO = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1.MP4"
DEFAULT_DATASET_DIR = SIM_ROOT / "2026" / "Prova_verisi_2026"
DEFAULT_REFERENCE_DIR = SIM_ROOT / "UAP_UAI_referans"
DEFAULT_REFERENCE_IMAGE = SIM_ROOT / "reference_objects" / "ref_01.jpg"
DEFAULT_REFERENCE_GT = DEFAULT_DATASET_DIR / "reference_objects_gt.json"
DEFAULT_OSTRACK_ROOT = SIM_ROOT / "trackers" / "OSTrack"
DEFAULT_OSTRACK_CHECKPOINT = (
    DEFAULT_OSTRACK_ROOT
    / "downloaded_models"
    / "models"
    / "vitb_256_mae_ce_32x4_ep300"
    / "OSTrack_ep0300.pth.tar"
)
DEFAULT_REFERENCE_TAG = "prova_v3_grand_finale"
DEFAULT_REFERENCE_PREDICTIONS = RESULTS_DIR / f"{DEFAULT_REFERENCE_TAG}_similarity_plain.json"
DEFAULT_HUMAN_MODEL = MODEL_DIR / "elcey.pt"
DEFAULT_VEHICLE_MODEL = MODEL_DIR / "vehicle.pt"
DEFAULT_LANDING_MODEL = MODEL_DIR / "UAP_UAI_V2.pt"
DEFAULT_CLASSIFIER = MODEL_DIR / "UAP_UAI_Classifier_resnet50_V4.1.pth"

CLASS_IDS = {
    "vehicle": "0",
    "human": "1",
    "UAP": "2",
    "UAI": "3",
}


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    cls: str
    landing_status: str
    motion_status: str
    score: float = 0.0
    name: str = ""


@dataclass
class VehicleTrack:
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    last_frame: int


@dataclass
class GroundTruth:
    bbox: tuple[int, int, int, int]
    cls: str
    category_name: str
    matched: bool = False


@dataclass
class ReferenceObjectFeatures:
    hist: np.ndarray
    edge: np.ndarray
    orb_keypoints: int
    orb_descriptors: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TEKNOFEST 2026 prova pipeline: frame bazli tespit, inis uygunlugu ve arac hareket durumu JSON ciktilari."
    )
    parser.add_argument("--source", choices=["dataset", "video"], default="dataset")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--annotations",
        type=Path,
        nargs="*",
        default=None,
        help="COCO annotation dosyalari. Bos ise dataset klasorundeki _annotations*.json dosyalari birlestirilir.",
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--human-model", type=Path, default=DEFAULT_HUMAN_MODEL)
    parser.add_argument("--vehicle-model", type=Path, default=DEFAULT_VEHICLE_MODEL)
    parser.add_argument("--landing-model", type=Path, default=DEFAULT_LANDING_MODEL)
    parser.add_argument("--landing-classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--device", type=str, default="0", help="CUDA cihaz id'si veya cpu.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.4, help="Eski komutlarla uyumluluk icin genel fallback conf.")
    parser.add_argument("--landing-conf", type=float, default=0.40, help="UAP/UAI tespit modeli icin conf.")
    parser.add_argument("--vehicle-conf", type=float, default=0.40, help="Arac tespit modeli icin conf.")
    parser.add_argument("--human-conf", type=float, default=0.25, help="Insan tespit modeli icin conf.")
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--max-frames", type=int, default=None, help="Kisa test icin islenecek maksimum frame.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--target-fps", type=float, default=None, help="Verilen FPS'e gore frame ornekle. Bos ise tum frame'ler.")
    parser.add_argument("--motion-threshold-px", type=float, default=12.0)
    parser.add_argument("--motion-edge-margin", type=int, default=2)
    parser.add_argument("--extra-nms-iou", type=float, default=0.75)
    parser.add_argument(
        "--cross-suppress-iou",
        type=float,
        default=0,
        help="Inis alaniyla neredeyse ayni kutudaki arac/insan yanlis pozitiflerini bastirma esigi.",
    )
    parser.add_argument("--classifier-positive-index", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--flip-landing-classes",
        action="store_true",
        help="Classifier sinif sirasi ters ise kullan: 0=uygun, 1=uygun degil.",
    )
    parser.add_argument("--save-annotated-video", action="store_true")
    parser.add_argument("--save-annotated-images", action="store_true")
    parser.add_argument("--save-frame-json", action="store_true", help="Her frame icin ayri JSON dosyasi yaz.")
    parser.add_argument("--score-iou", type=float, default=0.50)
    parser.add_argument("--task1-max-points", type=float, default=25.0, help="Birinci gorevin maksimum puani.")
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        default=DEFAULT_REFERENCE_PREDICTIONS,
        help="Sadece --no-run-reference-pipeline ile dis referans tahmin JSON'u okumak icin kullanilir.",
    )
    parser.add_argument(
        "--run-reference-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Once dahili OSTrack + similarity referans nesne hattini calistir.",
    )
    parser.add_argument("--reference-image", type=Path, default=DEFAULT_REFERENCE_IMAGE)
    parser.add_argument("--reference-gt", type=Path, default=DEFAULT_REFERENCE_GT)
    parser.add_argument("--ostrack-root", type=Path, default=DEFAULT_OSTRACK_ROOT)
    parser.add_argument("--ostrack-checkpoint", type=Path, default=DEFAULT_OSTRACK_CHECKPOINT)
    parser.add_argument("--reference-tag", default=DEFAULT_REFERENCE_TAG)
    parser.add_argument("--reference-limit", type=int, default=0)
    parser.add_argument("--reference-post-filter", choices=["adaptive", "similarity"], default="similarity")
    parser.add_argument("--reference-similarity-mode", choices=["plain", "sliding"], default="plain")
    parser.add_argument("--reference-deep-model", choices=["dinov2", "clip", "both", "none"], default="both")
    parser.add_argument("--reference-dinov2-repo", type=Path, default=None)
    parser.add_argument("--reference-dinov2-checkpoint", type=Path, default=None)
    parser.add_argument("--reference-accept-threshold", type=float, default=0.38)
    parser.add_argument("--reference-min-similarity", type=float, default=0.52)
    parser.add_argument("--reference-min-deep-score", type=float, default=0.72)
    parser.add_argument("--reference-min-ostrack-score", type=float, default=0.20)
    parser.add_argument("--reference-ostrack-floor", type=float, default=0.45)
    parser.add_argument("--reference-ostrack-cap", type=float, default=0.85)
    parser.add_argument("--reference-color-weight", type=float, default=0.45)
    parser.add_argument("--reference-edge-weight", type=float, default=0.35)
    parser.add_argument("--reference-corner-weight", type=float, default=0.20)
    parser.add_argument("--reference-deep-weight", type=float, default=0.60)
    parser.add_argument("--reference-lightglue-model", choices=["none", "superpoint"], default="superpoint")
    parser.add_argument("--reference-lightglue-weight", type=float, default=0.08)
    parser.add_argument("--reference-min-lightglue-score", type=float, default=0.20)
    parser.add_argument("--reference-require-lightglue", action="store_true")
    parser.add_argument("--reference-lightglue-max-keypoints", type=int, default=512)
    parser.add_argument("--reference-lightglue-min-matches", type=int, default=8)
    parser.add_argument("--reference-lightglue-match-norm", type=float, default=35.0)
    parser.add_argument("--reference-lightglue-min-pre-similarity", type=float, default=0.46)
    parser.add_argument("--reference-lightglue-min-deep-score", type=float, default=0.60)
    parser.add_argument("--reference-similarity-weight", type=float, default=0.58)
    parser.add_argument("--reference-sliding-top-k", type=int, default=3)
    parser.add_argument("--reference-percentile", type=float, default=0.95)
    parser.add_argument("--reference-mad-k", type=float, default=2.5)
    parser.add_argument("--reference-min-score", type=float, default=0.55)
    parser.add_argument("--reference-weak-ratio", type=float, default=0.88)
    parser.add_argument("--reference-max-gap", type=int, default=6)
    parser.add_argument("--reference-min-segment", type=int, default=3)
    parser.add_argument("--reference-tile-size", type=int, default=640)
    parser.add_argument("--reference-stride", type=int, default=320)
    parser.add_argument("--reference-candidates-per-frame", type=int, default=6)
    parser.add_argument("--reference-candidate-nms-iou", type=float, default=0.65)
    parser.add_argument("--reference-device", default="auto")
    parser.add_argument("--reference-object-id", default="ref_01", help="detected_undefined_objects icin object_id.")
    parser.add_argument("--reference-min-final-score", type=float, default=0.0)
    parser.add_argument("--no-reference-objects", action="store_true", help="Referans nesne eklemeyi kapat.")
    parser.add_argument(
        "--landing-embedding-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="UAP/UAI crop'larini DINOv2+CLIP referans benzerligiyle dogrula.",
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR, help="UAP/UAI referans crop klasoru.")
    parser.add_argument("--uap-reference", type=Path, nargs="*", default=[], help="UAP referans gorsel/crop dosyalari.")
    parser.add_argument("--uai-reference", type=Path, nargs="*", default=[], help="UAI referans gorsel/crop dosyalari.")
    parser.add_argument("--reference-samples-per-class", type=int, default=8)
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--dinov2-model", type=str, default="facebook/dinov2-base")
    parser.add_argument("--embedding-min-sim", type=float, default=0.18)
    parser.add_argument("--embedding-margin", type=float, default=0.025)
    parser.add_argument("--embedding-clip-weight", type=float, default=0.50)
    parser.add_argument("--embedding-dino-weight", type=float, default=0.50)
    return parser.parse_args()


def resolve_device(requested: str) -> tuple[str, torch.device]:
    if torch.cuda.is_available() and requested.lower() != "cpu":
        if requested.startswith("cuda:"):
            idx = requested.split(":", 1)[1]
        else:
            idx = requested
        return idx, torch.device(f"cuda:{idx}")
    return "cpu", torch.device("cpu")


def require_paths(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Bulunamayan dosya(lar):\n" + "\n".join(missing))


def load_landing_classifier(weights_path: Path, device: torch.device) -> torch.nn.Module:
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    state_dict = torch.load(weights_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def effective_model_conf(args: argparse.Namespace, model_name: str) -> float:
    model_conf = {
        "landing": args.landing_conf,
        "vehicle": args.vehicle_conf,
        "human": args.human_conf,
    }.get(model_name)
    return float(args.conf if model_conf is None else model_conf)


def yolo_predict(model: YOLO, frame: np.ndarray, args: argparse.Namespace, yolo_device: str, model_name: str) -> Any:
    return model.predict(
        source=frame,
        device=yolo_device,
        imgsz=args.imgsz,
        conf=effective_model_conf(args, model_name),
        iou=args.iou,
        max_det=args.max_det,
        half=(yolo_device != "cpu"),
        verbose=False,
    )[0]


def clamp_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(0, min(width - 1, int(round(x2))))
    bottom = max(0, min(height - 1, int(round(y2))))
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    return left, top, right, bottom


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def estimate_camera_affine(prev_frame: np.ndarray | None, frame: np.ndarray) -> np.ndarray:
    if prev_frame is None:
        return np.eye(2, 3, dtype=np.float32)

    scale = 0.5
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_small = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray_small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
    kp1, des1 = orb.detectAndCompute(prev_small, None)
    kp2, des2 = orb.detectAndCompute(gray_small, None)
    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return np.eye(2, 3, dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 12:
        return np.eye(2, 3, dtype=np.float32)

    matches = sorted(matches, key=lambda m: m.distance)[:250]
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if affine is None or inliers is None or int(inliers.sum()) < 8:
        return np.eye(2, 3, dtype=np.float32)

    affine = affine.astype(np.float32)
    affine[:, 2] /= scale
    return affine


def transform_point(point: tuple[float, float], affine: np.ndarray) -> tuple[float, float]:
    x, y = point
    return (
        float(affine[0, 0] * x + affine[0, 1] * y + affine[0, 2]),
        float(affine[1, 0] * x + affine[1, 1] * y + affine[1, 2]),
    )


def infer_vehicle_motion(
    vehicle_boxes: list[tuple[int, int, int, int]],
    prev_tracks: list[VehicleTrack],
    affine: np.ndarray,
    frame_idx: int,
    threshold_px: float,
) -> tuple[list[str], list[VehicleTrack]]:
    statuses: list[str] = []
    next_tracks: list[VehicleTrack] = []
    used_prev: set[int] = set()

    for box in vehicle_boxes:
        center = bbox_center(box)
        best_idx = -1
        best_score = -1.0
        best_residual = float("inf")

        for idx, track in enumerate(prev_tracks):
            if idx in used_prev:
                continue
            predicted = transform_point(track.center, affine)
            residual = math.dist(center, predicted)
            iou = bbox_iou(box, track.bbox)
            score = iou * 100.0 - residual
            if residual < max(80.0, threshold_px * 6.0) and score > best_score:
                best_idx = idx
                best_score = score
                best_residual = residual

        if best_idx >= 0:
            used_prev.add(best_idx)
            statuses.append("1" if best_residual > threshold_px else "0")
        else:
            statuses.append("0")

        next_tracks.append(VehicleTrack(center=center, bbox=box, last_frame=frame_idx))

    return statuses, next_tracks


def crop_with_padding(frame: np.ndarray, bbox: tuple[int, int, int, int], pad_ratio: float = 0.06) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1 + 1) * pad_ratio)
    pad_y = int((y2 - y1 + 1) * pad_ratio)
    x1, y1, x2, y2 = clamp_bbox(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w, h)
    return frame[y1 : y2 + 1, x1 : x2 + 1]


def bbox_touches_edge(bbox: tuple[int, int, int, int], width: int, height: int, margin: int) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= margin or y1 <= margin or x2 >= width - 1 - margin or y2 >= height - 1 - margin


def classify_landing_status(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    classifier: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    positive_index: int,
    edge_margin: int,
) -> str:
    height, width = frame.shape[:2]
    if bbox_touches_edge(bbox, width, height, edge_margin):
        return "0"

    crop = crop_with_padding(frame, bbox)
    if crop.size == 0:
        return "0"

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = classifier(tensor)
        pred_idx = int(torch.argmax(logits, dim=1).item())
    return "1" if pred_idx == positive_index else "0"


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_reference_images_from_paths(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Referans gorsel bulunamadi: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def load_reference_images_from_dir(reference_dir: Path) -> dict[str, list[Image.Image]]:
    references = {"UAP": [], "UAI": []}
    if not reference_dir.exists():
        return references

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in sorted(reference_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in image_extensions:
            continue
        name = path.stem.upper().replace("İ", "I")
        if "UAP" in name:
            references["UAP"].append(Image.open(path).convert("RGB"))
        elif "UAI" in name or "UAI" in name.replace("UAİ", "UAI"):
            references["UAI"].append(Image.open(path).convert("RGB"))
    return references


def load_reference_images_from_coco(
    dataset_dir: Path,
    annotation_paths: list[Path],
    samples_per_class: int,
) -> dict[str, list[Image.Image]]:
    references = {"UAP": [], "UAI": []}
    if samples_per_class <= 0:
        return references

    for ann_path in annotation_paths:
        if not ann_path.exists():
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        categories = {int(cat["id"]): str(cat["name"]).upper().replace("İ", "I") for cat in data.get("categories", [])}
        images = {int(img["id"]): img for img in data.get("images", [])}

        for ann in data.get("annotations", []):
            category = categories.get(int(ann["category_id"]), "")
            if category not in references or len(references[category]) >= samples_per_class:
                continue

            image_info = images.get(int(ann["image_id"]))
            if image_info is None:
                continue
            image_path = dataset_dir / image_info["file_name"]
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            x, y, w, h = ann["bbox"]
            bbox = clamp_bbox(x, y, x + w, y + h, frame.shape[1], frame.shape[0])
            crop = crop_with_padding(frame, bbox, pad_ratio=0.10)
            if crop.size:
                references[category].append(bgr_to_pil(crop))

            if all(len(items) >= samples_per_class for items in references.values()):
                return references

    return references


class LandingReferenceMatcher:
    def __init__(
        self,
        uap_references: list[Image.Image],
        uai_references: list[Image.Image],
        device: torch.device,
        clip_model_name: str,
        dinov2_model_name: str,
        min_similarity: float,
        margin: float,
        clip_weight: float,
        dino_weight: float,
    ) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "DINOv2+CLIP kontrolu icin transformers kurulu olmali. "
                "Kurulum: .venv\\Scripts\\python.exe -m pip install transformers"
            ) from exc

        if not uap_references or not uai_references:
            raise ValueError("UAP ve UAI icin en az birer referans gorsel gerekli.")

        total_weight = max(1e-9, clip_weight + dino_weight)
        self.clip_weight = clip_weight / total_weight
        self.dino_weight = dino_weight / total_weight
        self.min_similarity = min_similarity
        self.margin = margin
        self.device = device

        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        self.dino_model = AutoModel.from_pretrained(dinov2_model_name).to(device).eval()

        self.clip_refs = {
            CLASS_IDS["UAP"]: self._reference_centroid(uap_references, model_name="clip"),
            CLASS_IDS["UAI"]: self._reference_centroid(uai_references, model_name="clip"),
        }
        self.dino_refs = {
            CLASS_IDS["UAP"]: self._reference_centroid(uap_references, model_name="dino"),
            CLASS_IDS["UAI"]: self._reference_centroid(uai_references, model_name="dino"),
        }

    def classify(self, crop_bgr: np.ndarray) -> tuple[str | None, dict[str, float]]:
        if crop_bgr.size == 0:
            return None, {}

        image = bgr_to_pil(crop_bgr)
        clip_embedding = self._embed_clip(image)
        dino_embedding = self._embed_dino(image)
        scores: dict[str, float] = {}

        for cls_id in (CLASS_IDS["UAP"], CLASS_IDS["UAI"]):
            clip_sim = float(torch.sum(clip_embedding * self.clip_refs[cls_id]).item())
            dino_sim = float(torch.sum(dino_embedding * self.dino_refs[cls_id]).item())
            scores[f"{cls_id}_clip"] = clip_sim
            scores[f"{cls_id}_dino"] = dino_sim
            scores[cls_id] = self.clip_weight * clip_sim + self.dino_weight * dino_sim

        best_cls = max((CLASS_IDS["UAP"], CLASS_IDS["UAI"]), key=lambda cls_id: scores[cls_id])
        other_cls = CLASS_IDS["UAI"] if best_cls == CLASS_IDS["UAP"] else CLASS_IDS["UAP"]
        best_score = scores[best_cls]
        margin = best_score - scores[other_cls]
        scores["best_score"] = best_score
        scores["margin"] = margin

        if best_score < self.min_similarity or margin < self.margin:
            return None, scores
        return best_cls, scores

    def _reference_centroid(self, images: list[Image.Image], model_name: str) -> torch.Tensor:
        embeddings = [self._embed_clip(image) if model_name == "clip" else self._embed_dino(image) for image in images]
        centroid = torch.stack(embeddings, dim=0).mean(dim=0)
        return F.normalize(centroid, dim=0)

    def _embed_clip(self, image: Image.Image) -> torch.Tensor:
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.clip_model.get_image_features(**inputs)
            if isinstance(output, torch.Tensor):
                features = output.squeeze(0)
            elif hasattr(output, "image_embeds") and output.image_embeds is not None:
                features = output.image_embeds.squeeze(0)
            elif hasattr(output, "pooler_output") and output.pooler_output is not None:
                features = output.pooler_output.squeeze(0)
            else:
                features = output[0].squeeze(0)
        return F.normalize(features.float(), dim=0)

    def _embed_dino(self, image: Image.Image) -> torch.Tensor:
        inputs = self.dino_processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.dino_model(**inputs)
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                features = output.pooler_output.squeeze(0)
            else:
                features = output.last_hidden_state[:, 0].squeeze(0)
        return F.normalize(features.float(), dim=0)


def build_landing_reference_matcher(args: argparse.Namespace, device: torch.device) -> LandingReferenceMatcher | None:
    if not args.landing_embedding_check:
        return None

    annotation_paths = args.annotations if args.annotations else sorted(args.dataset_dir.glob("_annotations*.json"))
    dir_refs = load_reference_images_from_dir(args.reference_dir)
    auto_refs = {"UAP": [], "UAI": []}
    uap_refs = load_reference_images_from_paths(args.uap_reference) if args.uap_reference else dir_refs["UAP"]
    uai_refs = load_reference_images_from_paths(args.uai_reference) if args.uai_reference else dir_refs["UAI"]
    if not uap_refs or not uai_refs:
        auto_refs = load_reference_images_from_coco(args.dataset_dir, annotation_paths, args.reference_samples_per_class)
        if not uap_refs:
            uap_refs = auto_refs["UAP"]
        if not uai_refs:
            uai_refs = auto_refs["UAI"]

    print(f"[INFO] Referans klasoru: {args.reference_dir}")
    print(f"[INFO] UAP referans sayisi: {len(uap_refs)} | UAI referans sayisi: {len(uai_refs)}")
    return LandingReferenceMatcher(
        uap_references=uap_refs,
        uai_references=uai_refs,
        device=device,
        clip_model_name=args.clip_model,
        dinov2_model_name=args.dinov2_model,
        min_similarity=args.embedding_min_sim,
        margin=args.embedding_margin,
        clip_weight=args.embedding_clip_weight,
        dino_weight=args.embedding_dino_weight,
    )


def detections_from_yolo(result: Any, width: int, height: int) -> list[tuple[tuple[int, int, int, int], int, float, str]]:
    detections: list[tuple[tuple[int, int, int, int], int, float, str]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    names = result.names
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    for xyxy, cls_idx, score in zip(boxes, classes, scores):
        bbox = clamp_bbox(*xyxy, width, height)
        name = str(names.get(int(cls_idx), cls_idx)) if isinstance(names, dict) else str(cls_idx)
        detections.append((bbox, int(cls_idx), float(score), name))
    return detections


def nms_detections(
    detections: list[tuple[tuple[int, int, int, int], int, float, str]],
    iou_threshold: float,
) -> list[tuple[tuple[int, int, int, int], int, float, str]]:
    if not detections:
        return []

    kept: list[tuple[tuple[int, int, int, int], int, float, str]] = []
    for det in sorted(detections, key=lambda item: item[2], reverse=True):
        box, cls_idx, _score, _name = det
        if all(cls_idx != kept_det[1] or bbox_iou(box, kept_det[0]) < iou_threshold for kept_det in kept):
            kept.append(det)
    return kept


def suppress_boxes_overlapping_landing(
    detections: list[tuple[tuple[int, int, int, int], int, float, str]],
    landing_boxes: list[tuple[int, int, int, int]],
    iou_threshold: float,
) -> list[tuple[tuple[int, int, int, int], int, float, str]]:
    if not detections or not landing_boxes:
        return detections
    return [
        det
        for det in detections
        if all(bbox_iou(det[0], landing_box) < iou_threshold for landing_box in landing_boxes)
    ]


def make_detected_object(det: Detection) -> dict[str, str | int]:
    x1, y1, x2, y2 = det.bbox
    return {
        "cls": det.cls,
        "landing_status": det.landing_status,
        "motion_status": det.motion_status,
        "top_left_x": x1,
        "top_left_y": y1,
        "bottom_right_x": x2,
        "bottom_right_y": y2,
    }


def extract_frame_number(frame_name: str) -> int | None:
    stem = Path(frame_name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def make_reference_object(object_id: str, bbox: list[float] | tuple[float, float, float, float]) -> dict[str, str | int]:
    x1, y1, x2, y2 = bbox
    return {
        "object_id": object_id,
        "top_left_x": int(round(x1)),
        "top_left_y": int(round(y1)),
        "bottom_right_x": int(round(x2)),
        "bottom_right_y": int(round(y2)),
    }


def load_reference_predictions(
    path: Path,
    object_id: str,
    min_final_score: float = 0.0,
) -> dict[int, list[dict[str, Any]]]:
    if not path.exists():
        print(f"[WARN] Referans tahmin dosyasi bulunamadi: {path}")
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for item in data.get("frames", []):
        bbox = item.get("bbox")
        if not bbox:
            continue
        final_score = item.get("final_score", item.get("score", 1.0))
        if final_score is not None and float(final_score) < min_final_score:
            continue
        frame_number = extract_frame_number(str(item.get("frame_name", "")))
        if frame_number is None:
            continue
        by_frame.setdefault(frame_number, []).append(make_reference_object(object_id, bbox))
    return by_frame


def load_reference_predictions_from_payload(
    data: dict[str, Any],
    object_id: str,
    min_final_score: float = 0.0,
) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for item in data.get("frames", []):
        bbox = item.get("bbox")
        if not bbox:
            continue
        final_score = item.get("final_score", item.get("score", 1.0))
        if final_score is not None and float(final_score) < min_final_score:
            continue
        frame_number = extract_frame_number(str(item.get("frame_name", "")))
        if frame_number is None:
            continue
        by_frame.setdefault(frame_number, []).append(make_reference_object(object_id, bbox))
    return by_frame


def reference_bbox_from_object(item: dict[str, Any]) -> tuple[int, int, int, int] | None:
    if all(key in item for key in ("top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y")):
        return (
            int(round(float(item["top_left_x"]))),
            int(round(float(item["top_left_y"]))),
            int(round(float(item["bottom_right_x"]))),
            int(round(float(item["bottom_right_y"]))),
        )
    bbox = item.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return tuple(int(round(float(v))) for v in bbox)  # type: ignore[return-value]
    return None


def load_reference_ground_truths(path: Path) -> dict[int, list[dict[str, Any]]]:
    if not path.exists():
        print(f"[WARN] Referans GT dosyasi bulunamadi: {path}")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for frame in data.get("frames", []):
        frame_number = frame.get("frame_number")
        if frame_number is None:
            frame_number = extract_frame_number(str(frame.get("frame", frame.get("file_name", ""))))
        if frame_number is None:
            continue
        objects = []
        for obj in frame.get("objects", []):
            bbox = reference_bbox_from_object(obj)
            if bbox is not None:
                row = dict(obj)
                row["bbox_xyxy"] = bbox
                objects.append(row)
        by_frame[int(frame_number)] = objects
    return by_frame


def evaluate_reference_objects(
    predictions: dict[int, list[dict[str, Any]]],
    ground_truths: dict[int, list[dict[str, Any]]],
    frame_numbers: list[int],
    iou_threshold: float,
) -> dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_frames = 0
    rows = []

    for frame_number in frame_numbers:
        preds = [(item, reference_bbox_from_object(item)) for item in predictions.get(frame_number, [])]
        preds = [(item, bbox) for item, bbox in preds if bbox is not None]
        gts = [(item, item.get("bbox_xyxy")) for item in ground_truths.get(frame_number, []) if item.get("bbox_xyxy")]
        matched_gt: set[int] = set()
        frame_tp = 0
        frame_fp = 0

        for pred_item, pred_bbox in preds:
            assert pred_bbox is not None
            best_iou = 0.0
            best_index = -1
            for index, (_gt_item, gt_bbox) in enumerate(gts):
                if index in matched_gt:
                    continue
                iou = bbox_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index >= 0 and best_iou >= iou_threshold:
                matched_gt.add(best_index)
                frame_tp += 1
                rows.append(
                    {
                        "frame_number": frame_number,
                        "state": "tp",
                        "iou": best_iou,
                        "prediction": pred_item,
                        "ground_truth": gts[best_index][0],
                    }
                )
            else:
                frame_fp += 1
                rows.append({"frame_number": frame_number, "state": "fp", "iou": best_iou, "prediction": pred_item})

        frame_fn = max(0, len(gts) - len(matched_gt))
        if frame_tp > 0:
            matched_frames += 1
        total_tp += frame_tp
        total_fp += frame_fp
        total_fn += frame_fn
        if frame_fn:
            rows.append({"frame_number": frame_number, "state": "fn", "count": frame_fn})

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "iou_threshold": iou_threshold,
        "evaluated_frames": len(frame_numbers),
        "gt_object_frames": sum(1 for frame in frame_numbers if ground_truths.get(frame)),
        "matched_frames": matched_frames,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": rows,
    }


def reference_box(image: np.ndarray) -> list[float]:
    height, width = image.shape[:2]
    margin_x = int(width * 0.02)
    margin_y = int(height * 0.02)
    return [margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y]


def make_tiles(width: int, height: int, tile_size: int, stride: int):
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if not xs or xs[-1] != width - tile_size:
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] != height - tile_size:
        ys.append(max(0, height - tile_size))
    for y in ys:
        for x in xs:
            yield x, y, min(width, x + tile_size), min(height, y + tile_size)


def reference_candidate_images(args: argparse.Namespace) -> list[tuple[int, str, Path]]:
    if args.source == "dataset":
        image_paths, _ground_truths = load_coco_ground_truths(
            args.annotations if args.annotations else sorted(args.dataset_dir.glob("_annotations*.json")),
            args.dataset_dir,
        )
        if args.start_frame > 0:
            image_paths = image_paths[args.start_frame :]
        if args.max_frames is not None:
            image_paths = image_paths[: args.max_frames]
        if args.reference_limit and args.reference_limit > 0:
            image_paths = image_paths[: args.reference_limit]
        rows = []
        for path in image_paths:
            frame_id = canonical_frame_id(path.name, 0)
            frame_number = extract_frame_number(frame_id)
            if frame_number is not None:
                rows.append((frame_number, f"{frame_id}.jpg", path))
        return rows

    rows = []
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Video acilamadi: {args.video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start = max(0, args.start_frame)
    end = total if args.max_frames is None else min(total, start + args.max_frames)
    if args.reference_limit and args.reference_limit > 0:
        end = min(end, start + args.reference_limit)
    capture.release()
    for frame_number in range(start, end):
        rows.append((frame_number, f"frame_{frame_number:06d}.jpg", Path(f"__video_frame__:{frame_number}")))
    return rows


def read_reference_source_image(path: Path, args: argparse.Namespace) -> np.ndarray:
    marker = "__video_frame__:"
    if str(path).startswith(marker):
        frame_number = int(str(path).split(":", 1)[1])
        capture = cv2.VideoCapture(str(args.video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"Video frame okunamadi: {frame_number}")
        return frame
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def hsv_hist(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [36, 40], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist.astype(np.float32)


def edge_vector(image: np.ndarray, size: int = 128) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = min(size / max(1, width), size / max(1, height))
    resized = cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    edges = cv2.Canny(cv2.equalizeHist(canvas), 60, 160).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(edges))
    return edges / norm if norm > 1e-6 else edges


def crop_xyxy(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, width, height)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return image[y1 : y2 + 1, x1 : x2 + 1]


def reference_similarity_scores(ref_image: np.ndarray, crop: np.ndarray) -> tuple[float, float, float]:
    color_distance = float(cv2.compareHist(hsv_hist(ref_image), hsv_hist(crop), cv2.HISTCMP_BHATTACHARYYA))
    color = max(0.0, min(1.0, 1.0 - color_distance))
    edge = max(0.0, min(1.0, float(np.dot(edge_vector(ref_image), edge_vector(crop)))))
    similarity = 0.56 * color + 0.44 * edge
    return similarity, color, edge


def letterbox_rgb(image: np.ndarray, size: int = 224) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / max(1, width), size / max(1, height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 127, dtype=np.uint8)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def letterbox_gray(image: np.ndarray, size: int = 128) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = min(size / max(1, width), size / max(1, height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def reference_foreground_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if width < 20 or height < 20:
        return np.full((height, width), 255, dtype=np.uint8)

    mask = np.zeros((height, width), np.uint8)
    rect = (
        max(1, int(width * 0.04)),
        max(1, int(height * 0.04)),
        max(2, int(width * 0.92)),
        max(2, int(height * 0.92)),
    )
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    except cv2.error:
        mask = np.full((height, width), 255, dtype=np.uint8)

    area_ratio = float(np.count_nonzero(mask)) / float(width * height)
    if area_ratio < 0.03 or area_ratio > 0.96:
        mask = np.full((height, width), 255, dtype=np.uint8)
    else:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def reference_hsv_hist(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [36, 40], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist.astype(np.float32)


def reference_edge_vector(image: np.ndarray) -> np.ndarray:
    gray = letterbox_gray(image, 128)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    vec = edges.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-6 else vec


def reference_orb_features(image: np.ndarray, mask: np.ndarray | None) -> tuple[int, np.ndarray | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=900, scaleFactor=1.2, nlevels=8, edgeThreshold=12, patchSize=31)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    return len(keypoints), descriptors


def build_reference_object_features(reference: np.ndarray) -> ReferenceObjectFeatures:
    mask = reference_foreground_mask(reference)
    kp_count, descriptors = reference_orb_features(reference, mask)
    return ReferenceObjectFeatures(
        hist=reference_hsv_hist(reference, mask),
        edge=reference_edge_vector(reference),
        orb_keypoints=kp_count,
        orb_descriptors=descriptors,
    )


class ReferenceDeepEmbedder:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        clip_model: str,
        dinov2_repo: Path | None = None,
        dinov2_checkpoint: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model = None
        self.dino_model = None
        self.clip_model_obj = None
        self.processor = None
        self._pil_image_cls = Image

        if model_name == "none":
            return
        if model_name in {"dinov2", "both"}:
            if dinov2_repo and dinov2_repo.exists():
                if dinov2_checkpoint and dinov2_checkpoint.exists():
                    self.dino_model = torch.hub.load(
                        str(dinov2_repo),
                        "dinov2_vits14",
                        source="local",
                        weights=str(dinov2_checkpoint),
                    )
                else:
                    self.dino_model = torch.hub.load(str(dinov2_repo), "dinov2_vits14", source="local")
            else:
                self.dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.dino_model.to(self.device).eval()
            self.model = self.dino_model
            if model_name == "dinov2":
                return
        if model_name in {"clip", "both"}:
            try:
                from transformers import CLIPImageProcessor, CLIPModel
            except ImportError as exc:
                raise RuntimeError(
                    "CLIP icin transformers paketi gerekli. Ya --reference-deep-model dinov2/none kullan "
                    "ya da transformers kur."
                ) from exc

            self.clip_model_obj = CLIPModel.from_pretrained(clip_model).to(self.device).eval()
            self.processor = CLIPImageProcessor.from_pretrained(clip_model)
            self.model = self.model or self.clip_model_obj
            return
        raise ValueError(f"Bilinmeyen deep model: {model_name}")

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def _dinov2_tensor(self, image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = letterbox_rgb(rgb, 224)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device)

    def embed(self, image: np.ndarray) -> np.ndarray | None:
        if not self.enabled:
            return None
        assert self.model is not None
        with torch.inference_mode():
            parts = []
            if self.model_name in {"dinov2", "both"}:
                assert self.dino_model is not None
                dino_features = self.dino_model(self._dinov2_tensor(image))
                parts.append(F.normalize(dino_features.float(), dim=-1))
            if self.model_name in {"clip", "both"}:
                assert self.processor is not None
                assert self.clip_model_obj is not None
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = self._pil_image_cls.fromarray(rgb)
                inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
                clip_features = self.clip_model_obj.get_image_features(**inputs)
                if not torch.is_tensor(clip_features):
                    if hasattr(clip_features, "image_embeds"):
                        clip_features = clip_features.image_embeds
                    elif hasattr(clip_features, "pooler_output"):
                        clip_features = clip_features.pooler_output
                    elif hasattr(clip_features, "last_hidden_state"):
                        clip_features = clip_features.last_hidden_state[:, 0]
                    else:
                        raise TypeError(f"Beklenmeyen CLIP cikti tipi: {type(clip_features)!r}")
                parts.append(F.normalize(clip_features.float(), dim=-1))
            features = F.normalize(torch.cat(parts, dim=-1), dim=-1)
        return features.squeeze(0).detach().cpu().numpy().astype(np.float32)


class ReferenceLightGlueVerifier:
    def __init__(self, model_name: str, device: torch.device, max_keypoints: int) -> None:
        self.model_name = model_name
        self.device = device
        self.extractor = None
        self.matcher = None
        self.rbd = None
        self.ref_features = None

        if model_name == "none":
            return
        if model_name != "superpoint":
            raise ValueError(f"Bilinmeyen LightGlue modeli: {model_name}")
        try:
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import rbd
        except ImportError as exc:
            raise RuntimeError(
                "LightGlue kurulu degil. Kurulum: pip install git+https://github.com/cvg/LightGlue.git"
            ) from exc

        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        self.rbd = rbd

    @property
    def enabled(self) -> bool:
        return self.extractor is not None and self.matcher is not None

    def _tensor(self, image: np.ndarray, size: int = 512) -> torch.Tensor:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        height, width = gray.shape[:2]
        scale = min(size / max(1, width), size / max(1, height))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size), dtype=np.uint8)
        x = (size - new_w) // 2
        y = (size - new_h) // 2
        canvas[y : y + new_h, x : x + new_w] = resized
        return (torch.from_numpy(canvas).float()[None, None] / 255.0).to(self.device)

    def set_reference(self, reference: np.ndarray) -> None:
        if not self.enabled:
            return
        assert self.extractor is not None
        with torch.inference_mode():
            self.ref_features = self.extractor.extract(self._tensor(reference))

    def score(self, crop: np.ndarray, min_matches: int, match_norm: float) -> tuple[float, int, int]:
        if not self.enabled or self.ref_features is None:
            return 0.0, 0, 0
        assert self.extractor is not None and self.matcher is not None and self.rbd is not None
        with torch.inference_mode():
            crop_features = self.extractor.extract(self._tensor(crop))
            matches_payload = self.matcher({"image0": self.ref_features, "image1": crop_features})
            ref_features, crop_features, matches_payload = [
                self.rbd(item) for item in [self.ref_features, crop_features, matches_payload]
            ]

        matches = matches_payload["matches"]
        scores = matches_payload.get("scores")
        match_count = int(matches.shape[0])
        if match_count < min_matches:
            return 0.0, match_count, 0

        pts0 = ref_features["keypoints"][matches[:, 0]].detach().cpu().numpy()
        pts1 = crop_features["keypoints"][matches[:, 1]].detach().cpu().numpy()
        inliers = 0
        if match_count >= 4:
            _homography, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.ravel().sum())

        mean_conf = float(scores.mean().item()) if scores is not None and scores.numel() else 0.0
        inlier_ratio = inliers / float(match_count) if match_count else 0.0
        count_score = min(1.0, match_count / max(1.0, match_norm))
        score = 0.45 * inlier_ratio + 0.35 * count_score + 0.20 * mean_conf
        return max(0.0, min(1.0, score)), match_count, inliers


class DevicePreprocessor:
    def __init__(self, device: torch.device, nested_tensor_cls) -> None:
        self.device = device
        self.nested_tensor_cls = nested_tensor_cls
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, 3, 1, 1))
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        img_tensor = torch.as_tensor(img_arr, device=self.device).float().permute((2, 0, 1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).to(self.device).unsqueeze(dim=0)
        return self.nested_tensor_cls(img_tensor_norm, amask_tensor)


def reference_color_score(ref: ReferenceObjectFeatures, crop: np.ndarray) -> float:
    distance = float(cv2.compareHist(ref.hist, reference_hsv_hist(crop, None), cv2.HISTCMP_BHATTACHARYYA))
    return max(0.0, min(1.0, 1.0 - distance))


def reference_edge_score(ref: ReferenceObjectFeatures, crop: np.ndarray) -> float:
    return max(0.0, min(1.0, float(np.dot(ref.edge, reference_edge_vector(crop)))))


def reference_corner_score(ref: ReferenceObjectFeatures, crop: np.ndarray) -> float:
    if ref.orb_descriptors is None or ref.orb_keypoints < 6:
        return 0.0
    kp_count, descriptors = reference_orb_features(crop, None)
    if descriptors is None or kp_count < 4:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = matcher.knnMatch(ref.orb_descriptors, descriptors, k=2)
    except cv2.error:
        return 0.0

    good = 0
    for pair in matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.78 * second.distance:
            good += 1
    normalizer = max(10.0, math.sqrt(float(ref.orb_keypoints * max(1, kp_count))) * 0.18)
    return max(0.0, min(1.0, good / normalizer))


def reference_deep_score(ref_embedding: np.ndarray | None, embedder: ReferenceDeepEmbedder, crop: np.ndarray) -> float:
    if ref_embedding is None or not embedder.enabled:
        return 0.0
    crop_embedding = embedder.embed(crop)
    if crop_embedding is None:
        return 0.0
    cosine = float(np.dot(ref_embedding, crop_embedding))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def reference_ostrack_score_norm(score: float, floor: float = 0.45, cap: float = 0.85) -> float:
    return max(0.0, min(1.0, (score - floor) / (cap - floor))) if cap > floor else max(0.0, min(1.0, score))


def reference_box_iou_float(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def select_reference_ostrack_candidates(
    candidates: list[tuple[float, tuple[float, float, float, float]]],
    top_k: int,
    nms_iou: float,
) -> list[tuple[float, tuple[float, float, float, float]]]:
    selected: list[tuple[float, tuple[float, float, float, float]]] = []
    for score, box in sorted(candidates, key=lambda item: item[0], reverse=True):
        if all(reference_box_iou_float(box, kept_box) < nms_iou for _kept_score, kept_box in selected):
            selected.append((score, box))
        if len(selected) >= max(1, top_k):
            break
    return selected


def reference_candidate_windows(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    mode: str,
) -> list[tuple[float, float, float, float]]:
    clipped = (
        max(0.0, min(float(width), box[0])),
        max(0.0, min(float(height), box[1])),
        max(0.0, min(float(width), box[2])),
        max(0.0, min(float(height), box[3])),
    )
    if mode == "plain":
        return [clipped]

    x1, y1, x2, y2 = clipped
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(10.0, x2 - x1)
    bh = max(10.0, y2 - y1)
    scales = [0.82, 1.0, 1.18, 1.38]
    offsets = [-0.18, 0.0, 0.18]
    windows: list[tuple[float, float, float, float]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for scale in scales:
        sw = bw * scale
        sh = bh * scale
        for oy in offsets:
            for ox in offsets:
                ncx = cx + ox * bw
                ncy = cy + oy * bh
                candidate = (
                    max(0.0, min(float(width), ncx - sw / 2.0)),
                    max(0.0, min(float(height), ncy - sh / 2.0)),
                    max(0.0, min(float(width), ncx + sw / 2.0)),
                    max(0.0, min(float(height), ncy + sh / 2.0)),
                )
                key = tuple(int(round(v)) for v in candidate)
                if key not in seen:
                    seen.add(key)
                    windows.append(candidate)
    return windows


def score_reference_crop(
    args: argparse.Namespace,
    ref: ReferenceObjectFeatures,
    ref_embedding: np.ndarray | None,
    embedder: ReferenceDeepEmbedder,
    lightglue: ReferenceLightGlueVerifier,
    crop: np.ndarray,
    weights: tuple[float, float, float, float, float],
    include_corner: bool = True,
    include_deep: bool = True,
    include_lightglue: bool = True,
) -> tuple[float, float, float, float, float, float, int, int]:
    color = reference_color_score(ref, crop)
    edge = reference_edge_score(ref, crop)
    corner = reference_corner_score(ref, crop) if include_corner else 0.0
    deep = reference_deep_score(ref_embedding, embedder, crop) if include_deep else 0.0
    base_weight = max(1e-6, sum(weights[:4]))
    base_similarity = (weights[0] * color + weights[1] * edge + weights[2] * corner + weights[3] * deep) / base_weight

    lightglue_score = 0.0
    lightglue_matches = 0
    lightglue_inliers = 0
    should_run_lightglue = (
        include_lightglue
        and lightglue.enabled
        and base_similarity >= args.reference_lightglue_min_pre_similarity
        and (not embedder.enabled or deep >= args.reference_lightglue_min_deep_score)
    )
    if should_run_lightglue:
        lightglue_score, lightglue_matches, lightglue_inliers = lightglue.score(
            crop,
            args.reference_lightglue_min_matches,
            args.reference_lightglue_match_norm,
        )

    if should_run_lightglue:
        combined = (
            weights[0] * color
            + weights[1] * edge
            + weights[2] * corner
            + weights[3] * deep
            + weights[4] * lightglue_score
        )
    else:
        combined = base_similarity
    return combined, color, edge, corner, deep, lightglue_score, lightglue_matches, lightglue_inliers


def reference_quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(len(ordered) - 1, idx))]


def reference_robust_threshold(scores: list[float], percentile: float, mad_k: float, min_score: float) -> float:
    if not scores:
        return min_score
    median = statistics.median(scores)
    mad = statistics.median([abs(score - median) for score in scores]) or 1e-6
    return max(min_score, reference_quantile(scores, percentile), median + mad_k * mad)


def adaptive_filter_reference_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    scores = [float(row.get("score") or 0.0) for row in rows if row.get("bbox")]
    threshold = reference_robust_threshold(
        scores,
        args.reference_percentile,
        args.reference_mad_k,
        args.reference_min_score,
    )
    strong = [
        idx
        for idx, row in enumerate(rows)
        if row.get("bbox") and float(row.get("score") or 0.0) >= threshold
    ]
    keep: set[int] = set(strong)
    weak_threshold = threshold * args.reference_weak_ratio

    for left, right in zip(strong, strong[1:]):
        if 1 < right - left <= args.reference_max_gap:
            for idx in range(left, right + 1):
                if rows[idx].get("bbox") and float(rows[idx].get("score") or 0.0) >= weak_threshold:
                    keep.add(idx)

    segments: list[list[int]] = []
    current: list[int] = []
    for idx in sorted(keep):
        if not current or idx == current[-1] + 1:
            current.append(idx)
        else:
            segments.append(current)
            current = [idx]
    if current:
        segments.append(current)

    keep = {idx for segment in segments if len(segment) >= args.reference_min_segment for idx in segment}
    for idx, row in enumerate(rows):
        if idx in keep:
            row["state"] = "detected" if float(row.get("score") or 0.0) >= threshold else "bridged"
            row["final_score"] = float(row.get("score") or 0.0)
        else:
            row["bbox"] = None
            row["state"] = "empty"
            row["final_score"] = 0.0
    return rows, threshold


def cxcywh_to_xyxy_ref(box: list[float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = box
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def run_reference_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if not args.reference_image.exists():
        raise FileNotFoundError(f"Referans gorsel bulunamadi: {args.reference_image}")
    if not args.ostrack_root.exists():
        raise FileNotFoundError(f"OSTrack klasoru bulunamadi: {args.ostrack_root}")
    if not args.ostrack_checkpoint.exists():
        raise FileNotFoundError(f"OSTrack checkpoint bulunamadi: {args.ostrack_checkpoint}")

    print("[INFO] Dahili referans pipeline calisiyor (OSTrack + similarity).", flush=True)
    sys.path.insert(0, str(args.ostrack_root))
    from lib.config.ostrack.config import cfg, update_config_from_file
    from lib.models.ostrack import build_ostrack
    from lib.train.data.processing_utils import sample_target
    from lib.utils.misc import NestedTensor

    update_config_from_file(str(args.ostrack_root / "experiments" / "ostrack" / "vitb_256_mae_ce_32x4_ep300.yaml"))
    device_request = args.device if args.reference_device == "auto" else args.reference_device
    _reference_device, torch_device = resolve_device(device_request)
    network = build_ostrack(cfg, training=False)
    checkpoint = torch.load(str(args.ostrack_checkpoint), map_location="cpu")
    network.load_state_dict(checkpoint["net"], strict=True)
    network.to(torch_device).eval()
    preprocessor = DevicePreprocessor(torch_device, NestedTensor)

    ref_bgr = cv2.imread(str(args.reference_image), cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise FileNotFoundError(args.reference_image)
    ref_features = None
    embedder = None
    ref_embedding = None
    lightglue = ReferenceLightGlueVerifier("none", torch_device, args.reference_lightglue_max_keypoints)
    sim_weights = None
    if args.reference_post_filter == "similarity":
        ref_features = build_reference_object_features(ref_bgr)
        embedder = ReferenceDeepEmbedder(
            args.reference_deep_model,
            torch_device,
            args.clip_model,
            args.reference_dinov2_repo,
            args.reference_dinov2_checkpoint,
        )
        ref_embedding = embedder.embed(ref_bgr) if embedder.enabled else None
        lightglue = ReferenceLightGlueVerifier(
            args.reference_lightglue_model,
            torch_device,
            args.reference_lightglue_max_keypoints,
        )
        lightglue.set_reference(ref_bgr)
        deep_weight = args.reference_deep_weight if embedder.enabled else 0.0
        lightglue_weight = args.reference_lightglue_weight if lightglue.enabled else 0.0
        raw_weight_sum = max(
            1e-6,
            args.reference_color_weight
            + args.reference_edge_weight
            + args.reference_corner_weight
            + deep_weight
            + lightglue_weight,
        )
        sim_weights = (
            args.reference_color_weight / raw_weight_sum,
            args.reference_edge_weight / raw_weight_sum,
            args.reference_corner_weight / raw_weight_sum,
            deep_weight / raw_weight_sum,
            lightglue_weight / raw_weight_sum,
        )
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    z_patch, _resize_factor, z_mask = sample_target(
        ref_rgb,
        reference_box(ref_rgb),
        cfg.TEST.TEMPLATE_FACTOR,
        output_sz=cfg.TEST.TEMPLATE_SIZE,
    )
    template = preprocessor.process(z_patch, z_mask)

    frames = []
    rows = reference_candidate_images(args)
    with torch.inference_mode():
        for index, (frame_number, frame_name, path) in enumerate(rows, start=1):
            frame_bgr = read_reference_source_image(path, args)
            image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            best = None
            tile_candidates: list[tuple[float, tuple[float, float, float, float]]] = []
            for x1, y1, x2, y2 in make_tiles(width, height, args.reference_tile_size, args.reference_stride):
                tile = image[y1:y2, x1:x2, :]
                tile_resized = cv2.resize(tile, (cfg.TEST.SEARCH_SIZE, cfg.TEST.SEARCH_SIZE))
                tile_mask = np.zeros((cfg.TEST.SEARCH_SIZE, cfg.TEST.SEARCH_SIZE), dtype=np.bool_)
                search = preprocessor.process(tile_resized, tile_mask)
                out = network.forward(template=template.tensors, search=search.tensors, ce_template_mask=None)
                score_map = out["score_map"]
                score = float(score_map.max().item())
                pred_boxes = network.box_head.cal_bbox(score_map, out["size_map"], out["offset_map"]).view(-1, 4)
                pred = [v * args.reference_tile_size for v in pred_boxes.mean(dim=0).detach().cpu().tolist()]
                bx1, by1, bx2, by2 = cxcywh_to_xyxy_ref(pred)
                box = (
                    max(0.0, min(float(width), bx1 + x1)),
                    max(0.0, min(float(height), by1 + y1)),
                    max(0.0, min(float(width), bx2 + x1)),
                    max(0.0, min(float(height), by2 + y1)),
                )
                tile_candidates.append((score, box))
                if best is None or score > best[0]:
                    best = (score, box)

            if args.reference_post_filter == "similarity":
                selected_candidates = select_reference_ostrack_candidates(
                    [
                        (score, box)
                        for score, box in tile_candidates
                        if score >= args.reference_min_ostrack_score
                    ],
                    args.reference_candidates_per_frame,
                    args.reference_candidate_nms_iou,
                )
            else:
                selected_candidates = [best] if best else []

            out_row: dict[str, Any] = {
                "frame_name": frame_name,
                "bbox": None,
                "score": float(best[0]) if best else 0.0,
                "final_score": 0.0,
                "similarity": 0.0,
                "color_score": 0.0,
                "edge_score": 0.0,
                "corner_score": 0.0,
                "deep_score": 0.0,
                "lightglue_score": 0.0,
                "lightglue_matches": 0,
                "lightglue_inliers": 0,
                "state": "empty",
            }
            if args.reference_post_filter == "adaptive":
                raw_score, raw_box = best if best else (0.0, None)
                if raw_box and raw_score >= args.reference_min_ostrack_score:
                    out_row["bbox"] = [float(v) for v in raw_box]
                    out_row["final_score"] = raw_score
                    out_row["state"] = "candidate"
            else:
                assert ref_features is not None and embedder is not None and sim_weights is not None
                best_candidate_row = None
                diagnostics_seen = 0
                for candidate_rank, (raw_score, raw_box) in enumerate(selected_candidates):
                    scored_windows = []
                    best_window = None
                    for window in reference_candidate_windows(raw_box, width, height, args.reference_similarity_mode):
                        crop = crop_xyxy(frame_bgr, window)
                        if crop is None:
                            continue
                        include_expensive = args.reference_similarity_mode == "plain"
                        (
                            similarity,
                            color,
                            edge,
                            corner,
                            deep,
                            lightglue_score,
                            lightglue_matches,
                            lightglue_inliers,
                        ) = score_reference_crop(
                            args,
                            ref_features,
                            ref_embedding,
                            embedder,
                            lightglue,
                            crop,
                            sim_weights,
                            include_corner=include_expensive,
                            include_deep=include_expensive,
                            include_lightglue=include_expensive,
                        )
                        norm_score = reference_ostrack_score_norm(
                            raw_score,
                            args.reference_ostrack_floor,
                            args.reference_ostrack_cap,
                        )
                        final_score = (
                            args.reference_similarity_weight * similarity
                            + (1.0 - args.reference_similarity_weight) * norm_score
                        )
                        scored_windows.append(
                            (
                                final_score,
                                similarity,
                                color,
                                edge,
                                corner,
                                deep,
                                lightglue_score,
                                lightglue_matches,
                                lightglue_inliers,
                                window,
                                crop,
                            )
                        )
                    if args.reference_similarity_mode == "sliding":
                        top_windows = sorted(scored_windows, key=lambda item: item[0], reverse=True)[
                            : max(1, args.reference_sliding_top_k)
                        ]
                        rescored_windows = []
                        for _old_final, _old_sim, _old_color, _old_edge, _old_corner, _old_deep, _old_lg, _old_lgm, _old_lgi, window, crop in top_windows:
                            (
                                similarity,
                                color,
                                edge,
                                corner,
                                deep,
                                lightglue_score,
                                lightglue_matches,
                                lightglue_inliers,
                            ) = score_reference_crop(
                                args,
                                ref_features,
                                ref_embedding,
                                embedder,
                                lightglue,
                                crop,
                                sim_weights,
                                include_corner=True,
                                include_deep=True,
                                include_lightglue=True,
                            )
                            norm_score = reference_ostrack_score_norm(
                                raw_score,
                                args.reference_ostrack_floor,
                                args.reference_ostrack_cap,
                            )
                            final_score = (
                                args.reference_similarity_weight * similarity
                                + (1.0 - args.reference_similarity_weight) * norm_score
                            )
                            rescored_windows.append(
                                (
                                    final_score,
                                    similarity,
                                    color,
                                    edge,
                                    corner,
                                    deep,
                                    lightglue_score,
                                    lightglue_matches,
                                    lightglue_inliers,
                                    window,
                                    crop,
                                )
                        )
                        scored_windows = rescored_windows
                    if scored_windows:
                        best_window = max(scored_windows, key=lambda item: item[0])
                    if best_window is not None:
                        (
                            final_score,
                            similarity,
                            color,
                            edge,
                            corner,
                            deep,
                            lightglue_score,
                            lightglue_matches,
                            lightglue_inliers,
                            box,
                            _crop,
                        ) = best_window
                        candidate_row = {
                            "frame_name": frame_name,
                            "bbox": None,
                            "score": raw_score,
                            "candidate_rank": candidate_rank,
                            "final_score": final_score,
                            "similarity": similarity,
                            "color_score": color,
                            "edge_score": edge,
                            "corner_score": corner,
                            "deep_score": deep,
                            "lightglue_score": lightglue_score,
                            "lightglue_matches": lightglue_matches,
                            "lightglue_inliers": lightglue_inliers,
                            "state": "rejected",
                        }
                        deep_ok = not embedder.enabled or deep >= args.reference_min_deep_score
                        lightglue_ok = (
                            not lightglue.enabled
                            or lightglue_score >= args.reference_min_lightglue_score
                            or (lightglue_matches == 0 and not args.reference_require_lightglue)
                        )
                        if (
                            final_score >= args.reference_accept_threshold
                            and similarity >= args.reference_min_similarity
                            and deep_ok
                            and lightglue_ok
                        ):
                            candidate_row["bbox"] = [float(v) for v in box]
                            candidate_row["state"] = "detected"
                        if best_candidate_row is None or float(candidate_row["final_score"]) > float(best_candidate_row["final_score"]):
                            best_candidate_row = candidate_row
                        if candidate_row["state"] == "detected" and (
                            out_row["bbox"] is None or float(candidate_row["final_score"]) > float(out_row["final_score"])
                        ):
                            out_row = candidate_row
                        diagnostics_seen += 1
                if out_row["bbox"] is None and best_candidate_row is not None:
                    out_row = best_candidate_row
                out_row["candidate_count"] = len(selected_candidates)
                out_row["diagnostics_seen"] = diagnostics_seen
            frames.append(out_row)
            if index % 50 == 0:
                print(f"[INFO] Referans frame: {index}/{len(rows)}", flush=True)

    filter_info = None
    if args.reference_post_filter == "adaptive":
        frames, threshold = adaptive_filter_reference_rows(args, frames)
        filter_info = {
            "threshold": threshold,
            "percentile": args.reference_percentile,
            "mad_k": args.reference_mad_k,
            "min_score": args.reference_min_score,
            "weak_ratio": args.reference_weak_ratio,
            "max_gap": args.reference_max_gap,
            "min_segment": args.reference_min_segment,
        }

    payload = {
        "reference": str(args.reference_image),
        "mode": f"internal_ostrack_{args.reference_post_filter}",
        "filter": filter_info,
        "deep_model": args.reference_deep_model if embedder and embedder.enabled else "none",
        "lightglue_model": args.reference_lightglue_model if lightglue.enabled else "none",
        "scoring": {
            "similarity_mode": args.reference_similarity_mode,
            "accept_threshold": args.reference_accept_threshold,
            "min_similarity": args.reference_min_similarity,
            "min_deep_score": args.reference_min_deep_score,
            "min_ostrack_score": args.reference_min_ostrack_score,
            "ostrack_floor": args.reference_ostrack_floor,
            "ostrack_cap": args.reference_ostrack_cap,
            "similarity_weight": args.reference_similarity_weight,
            "color_weight": args.reference_color_weight,
            "edge_weight": args.reference_edge_weight,
            "corner_weight": args.reference_corner_weight,
            "deep_weight": args.reference_deep_weight,
            "lightglue_weight": args.reference_lightglue_weight,
            "min_lightglue_score": args.reference_min_lightglue_score,
            "require_lightglue": args.reference_require_lightglue,
        },
        "frames": frames,
    }
    print(f"[INFO] Referans nesne: detected={sum(1 for f in frames if f.get('bbox'))}/{len(frames)}")
    return payload


def make_frame_packet(
    frame_idx: int,
    detections: list[Detection],
    frame_id: str | None = None,
    reference_objects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    frame_id = frame_id or f"frame_{frame_idx:06d}"
    return {
        "id": f"prediction_{frame_idx:06d}",
        "user": "",
        "frame": frame_id,
        "detected_objects": [make_detected_object(det) for det in detections],
        "detected_translations": [],
        "detected_undefined_objects": reference_objects or [],
    }


def write_predictions_json(path: Path, packets: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_annotations(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = frame.copy()
    colors = {
        "0": (0, 180, 255),
        "1": (255, 180, 0),
        "2": (80, 220, 80),
        "3": (220, 80, 220),
    }
    names = {"0": "vehicle", "1": "human", "2": "UAP", "3": "UAI"}
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = colors.get(det.cls, (255, 255, 255))
        label = f"{names.get(det.cls, det.cls)} L:{det.landing_status} M:{det.motion_status}"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output


def draw_ground_truth(frame: np.ndarray, ground_truths: list[GroundTruth]) -> np.ndarray:
    output = frame.copy()
    colors = {
        "0": (0, 120, 255),
        "1": (255, 120, 0),
        "2": (0, 210, 0),
        "3": (210, 0, 210),
    }
    for gt in ground_truths:
        x1, y1, x2, y2 = gt.bbox
        color = colors.get(gt.cls, (255, 255, 255))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        cv2.putText(output, f"GT {gt.category_name}", (x1, min(output.shape[0] - 6, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return output


def open_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter acilamadi: {path}")
    return writer


def should_process_frame(frame_idx: int, source_fps: float, target_fps: float | None) -> bool:
    if target_fps is None or target_fps <= 0 or source_fps <= 0 or target_fps >= source_fps:
        return True
    interval = source_fps / target_fps
    return abs((frame_idx / interval) - round(frame_idx / interval)) < (0.5 / interval)


def canonical_frame_id(file_name: str, fallback_idx: int) -> str:
    stem = Path(file_name).stem
    if "_jpg.rf." in stem:
        stem = stem.split("_jpg.rf.", 1)[0]
    return stem or f"frame_{fallback_idx:06d}"


def frame_sort_key(path: Path) -> tuple[int, str]:
    stem = canonical_frame_id(path.name, 0)
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (int(digits) if digits else 10**12, path.name)


def load_coco_ground_truths(annotation_paths: list[Path], dataset_dir: Path) -> tuple[list[Path], dict[str, list[GroundTruth]]]:
    coco_to_spec = {
        "tasit": CLASS_IDS["vehicle"],
        "taşıt": CLASS_IDS["vehicle"],
        "vehicle": CLASS_IDS["vehicle"],
        "insan": CLASS_IDS["human"],
        "human": CLASS_IDS["human"],
        "uap": CLASS_IDS["UAP"],
        "uai": CLASS_IDS["UAI"],
        "uai̇": CLASS_IDS["UAI"],
        "uaı": CLASS_IDS["UAI"],
    }
    image_paths: dict[str, Path] = {}
    truths: dict[str, list[GroundTruth]] = {}

    for ann_path in annotation_paths:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        categories = {int(cat["id"]): str(cat["name"]) for cat in data.get("categories", [])}
        images = {int(img["id"]): img for img in data.get("images", [])}

        for img in images.values():
            file_name = img["file_name"]
            path = dataset_dir / file_name
            if path.exists():
                image_paths[file_name] = path
                truths.setdefault(file_name, [])

        for ann in data.get("annotations", []):
            image = images.get(int(ann["image_id"]))
            if image is None:
                continue
            file_name = image["file_name"]
            path = dataset_dir / file_name
            if not path.exists():
                continue

            category_name = categories.get(int(ann["category_id"]), "")
            spec_cls = coco_to_spec.get(category_name.lower())
            if spec_cls is None:
                continue

            x, y, w, h = ann["bbox"]
            width = int(image.get("width", 1920))
            height = int(image.get("height", 1080))
            bbox = clamp_bbox(x, y, x + w, y + h, width, height)
            truths.setdefault(file_name, []).append(GroundTruth(bbox=bbox, cls=spec_cls, category_name=category_name))

    return sorted(image_paths.values(), key=frame_sort_key), truths


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    ap = 0.0
    for idx in range(1, len(mrec)):
        if mrec[idx] != mrec[idx - 1]:
            ap += (mrec[idx] - mrec[idx - 1]) * mpre[idx]
    return ap


def evaluate_predictions(
    predictions: dict[str, list[Detection]],
    ground_truths: dict[str, list[GroundTruth]],
    iou_threshold: float,
) -> dict[str, Any]:
    class_names = {"0": "tasit", "1": "insan", "2": "UAP", "3": "UAI"}
    summary: dict[str, Any] = {"iou_threshold": iou_threshold, "classes": {}}
    aps: list[float] = []
    total_tp = total_fp = total_fn = 0

    for cls_id, cls_name in class_names.items():
        gt_by_image = {
            file_name: [GroundTruth(bbox=gt.bbox, cls=gt.cls, category_name=gt.category_name) for gt in gts if gt.cls == cls_id]
            for file_name, gts in ground_truths.items()
        }
        gt_count = sum(len(gts) for gts in gt_by_image.values())
        pred_items: list[tuple[str, Detection]] = []
        for file_name, dets in predictions.items():
            pred_items.extend((file_name, det) for det in dets if det.cls == cls_id)
        pred_items.sort(key=lambda item: item[1].score, reverse=True)

        tps: list[int] = []
        fps: list[int] = []
        match_rows: list[dict[str, Any]] = []

        for file_name, det in pred_items:
            candidates = gt_by_image.get(file_name, [])
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(candidates):
                if gt.matched:
                    continue
                iou = bbox_iou(det.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0 and best_iou >= iou_threshold:
                candidates[best_idx].matched = True
                tps.append(1)
                fps.append(0)
                verdict = "TP"
            else:
                tps.append(0)
                fps.append(1)
                verdict = "FP"

            match_rows.append(
                {
                    "image": file_name,
                    "class": cls_name,
                    "score": det.score,
                    "iou": best_iou,
                    "verdict": verdict,
                    "bbox": list(det.bbox),
                }
            )

        cum_tp = np.cumsum(tps) if tps else np.array([])
        cum_fp = np.cumsum(fps) if fps else np.array([])
        recalls = (cum_tp / max(1, gt_count)).tolist() if len(cum_tp) else []
        precisions = (cum_tp / np.maximum(1, cum_tp + cum_fp)).tolist() if len(cum_tp) else []
        ap = average_precision(recalls, precisions)
        tp = int(cum_tp[-1]) if len(cum_tp) else 0
        fp = int(cum_fp[-1]) if len(cum_fp) else 0
        fn = gt_count - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, gt_count)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)

        if gt_count > 0:
            aps.append(ap)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        summary["classes"][cls_id] = {
            "name": cls_name,
            "ground_truth": gt_count,
            "predictions": len(pred_items),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": ap,
            "matches": match_rows,
        }

    micro_precision = total_tp / max(1, total_tp + total_fp)
    micro_recall = total_tp / max(1, total_tp + total_fn)
    summary["overall"] = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": 2 * micro_precision * micro_recall / max(1e-9, micro_precision + micro_recall),
        "map50": float(sum(aps) / max(1, len(aps))),
    }
    return summary


def predict_frame(
    frame: np.ndarray,
    frame_idx: int,
    args: argparse.Namespace,
    yolo_device: str,
    torch_device: torch.device,
    landing_detector: YOLO,
    vehicle_detector: YOLO,
    human_detector: YOLO,
    landing_classifier: torch.nn.Module,
    classifier_transform: transforms.Compose,
    positive_index: int,
    landing_matcher: LandingReferenceMatcher | None,
    prev_frame: np.ndarray | None,
    prev_tracks: list[VehicleTrack],
) -> tuple[list[Detection], list[VehicleTrack]]:
    height, width = frame.shape[:2]
    camera_affine = estimate_camera_affine(prev_frame, frame)
    detections: list[Detection] = []

    landing_result = yolo_predict(landing_detector, frame, args, yolo_device, "landing")
    landing_yolo_detections = nms_detections(
        detections_from_yolo(landing_result, width, height),
        args.extra_nms_iou,
    )
    for bbox, _cls_idx, score, name in landing_yolo_detections:
        yolo_name = name.upper().replace("İ", "I").replace("Ä°", "I")
        spec_cls = CLASS_IDS["UAI"] if yolo_name == "UAI" else CLASS_IDS["UAP"]
        if landing_matcher is not None:
            ref_cls, ref_scores = landing_matcher.classify(crop_with_padding(frame, bbox, pad_ratio=0.10))
            if ref_cls is None:
                continue
            spec_cls = ref_cls
            best_ref_score = ref_scores.get("best_score", score)
            score = float((score + best_ref_score) / 2.0)
            name = f"{name}|embed:{'UAI' if spec_cls == CLASS_IDS['UAI'] else 'UAP'}"
        landing_status = classify_landing_status(
            frame,
            bbox,
            landing_classifier,
            classifier_transform,
            torch_device,
            positive_index,
            args.motion_edge_margin,
        )
        detections.append(
            Detection(
                bbox=bbox,
                cls=spec_cls,
                landing_status=landing_status,
                motion_status="-1",
                score=score,
                name=name,
            )
        )

    vehicle_result = yolo_predict(vehicle_detector, frame, args, yolo_device, "vehicle")
    landing_boxes = [det.bbox for det in detections if det.cls in (CLASS_IDS["UAP"], CLASS_IDS["UAI"])]
    vehicle_yolo_detections = nms_detections(
        suppress_boxes_overlapping_landing(
            detections_from_yolo(vehicle_result, width, height),
            landing_boxes,
            args.cross_suppress_iou,
        ),
        args.extra_nms_iou,
    )
    vehicle_boxes = [bbox for bbox, _cls_idx, _score, _name in vehicle_yolo_detections]
    vehicle_scores = [score for _bbox, _cls_idx, score, _name in vehicle_yolo_detections]
    motion_statuses, next_tracks = infer_vehicle_motion(
        vehicle_boxes,
        prev_tracks,
        camera_affine,
        frame_idx,
        args.motion_threshold_px,
    )
    for bbox, score, motion_status in zip(vehicle_boxes, vehicle_scores, motion_statuses):
        detections.append(
            Detection(
                bbox=bbox,
                cls=CLASS_IDS["vehicle"],
                landing_status="-1",
                motion_status=motion_status,
                score=score,
                name="vehicle",
            )
        )

    human_result = yolo_predict(human_detector, frame, args, yolo_device, "human")
    human_yolo_detections = nms_detections(
        suppress_boxes_overlapping_landing(
            detections_from_yolo(human_result, width, height),
            landing_boxes,
            args.cross_suppress_iou,
        ),
        args.extra_nms_iou,
    )
    for bbox, _cls_idx, score, name in human_yolo_detections:
        detections.append(
            Detection(
                bbox=bbox,
                cls=CLASS_IDS["human"],
                landing_status="-1",
                motion_status="-1",
                score=score,
                name=name,
            )
        )

    return detections, next_tracks


def main() -> None:
    args = parse_args()
    require_paths([args.video, args.human_model, args.vehicle_model, args.landing_model, args.landing_classifier])

    args.results_dir.mkdir(parents=True, exist_ok=True)
    frame_json_dir = args.results_dir / "frame_json"
    if args.save_frame_json:
        frame_json_dir.mkdir(parents=True, exist_ok=True)

    yolo_device, torch_device = resolve_device(args.device)
    if args.flip_landing_classes:
        positive_index = 0
    else:
        positive_index = args.classifier_positive_index

    print(f"[INFO] Video: {args.video}")
    print(f"[INFO] Results: {args.results_dir}")
    print(f"[INFO] Device: cuda:{yolo_device}" if yolo_device != "cpu" else "[INFO] Device: cpu")
    print("[INFO] Modeller yukleniyor...")

    landing_detector = YOLO(str(args.landing_model))
    vehicle_detector = YOLO(str(args.vehicle_model))
    human_detector = YOLO(str(args.human_model))
    landing_classifier = load_landing_classifier(args.landing_classifier, torch_device)
    classifier_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Video acilamadi: {args.video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = args.target_fps if args.target_fps and args.target_fps > 0 else (source_fps or 25.0)

    if args.start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    annotated_writer = None
    if args.save_annotated_video:
        annotated_writer = open_video_writer(args.results_dir / "annotated_prova.mp4", output_fps, width, height)

    jsonl_path = args.results_dir / "predictions.jsonl"
    summary_path = args.results_dir / "run_summary.json"
    processed_count = 0
    seen_count = 0
    prev_frame: np.ndarray | None = None
    prev_tracks: list[VehicleTrack] = []
    started_at = time.time()

    print(f"[INFO] FPS: {source_fps:.3f} | Cozunurluk: {width}x{height} | Toplam frame: {total_frames}")
    print(f"[INFO] JSONL cikti: {jsonl_path}")

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_idx = args.start_frame + seen_count
            seen_count += 1
            if not should_process_frame(frame_idx, source_fps, args.target_fps):
                continue

            camera_affine = estimate_camera_affine(prev_frame, frame)
            detections: list[Detection] = []

            landing_result = yolo_predict(landing_detector, frame, args, yolo_device, "landing")
            landing_yolo_detections = nms_detections(
                detections_from_yolo(landing_result, width, height),
                args.extra_nms_iou,
            )
            for bbox, cls_idx, score, name in landing_yolo_detections:
                yolo_name = name.upper().replace("İ", "I")
                spec_cls = CLASS_IDS["UAI"] if yolo_name == "UAI" else CLASS_IDS["UAP"]
                landing_status = classify_landing_status(
                    frame,
                    bbox,
                    landing_classifier,
                    classifier_transform,
                    torch_device,
                    positive_index,
                    args.motion_edge_margin,
                )
                detections.append(
                    Detection(
                        bbox=bbox,
                        cls=spec_cls,
                        landing_status=landing_status,
                        motion_status="-1",
                        score=score,
                        name=name,
                    )
                )

            vehicle_result = yolo_predict(vehicle_detector, frame, args, yolo_device, "vehicle")
            landing_boxes = [det.bbox for det in detections if det.cls in (CLASS_IDS["UAP"], CLASS_IDS["UAI"])]
            vehicle_yolo_detections = nms_detections(
                suppress_boxes_overlapping_landing(
                    detections_from_yolo(vehicle_result, width, height),
                    landing_boxes,
                    args.cross_suppress_iou,
                ),
                args.extra_nms_iou,
            )
            vehicle_boxes = [bbox for bbox, _cls_idx, _score, _name in vehicle_yolo_detections]
            vehicle_scores = [score for _bbox, _cls_idx, score, _name in vehicle_yolo_detections]
            motion_statuses, prev_tracks = infer_vehicle_motion(
                vehicle_boxes,
                prev_tracks,
                camera_affine,
                frame_idx,
                args.motion_threshold_px,
            )
            for bbox, score, motion_status in zip(vehicle_boxes, vehicle_scores, motion_statuses):
                detections.append(
                    Detection(
                        bbox=bbox,
                        cls=CLASS_IDS["vehicle"],
                        landing_status="-1",
                        motion_status=motion_status,
                        score=score,
                        name="vehicle",
                    )
                )

            human_result = yolo_predict(human_detector, frame, args, yolo_device, "human")
            human_yolo_detections = nms_detections(
                suppress_boxes_overlapping_landing(
                    detections_from_yolo(human_result, width, height),
                    landing_boxes,
                    args.cross_suppress_iou,
                ),
                args.extra_nms_iou,
            )
            for bbox, _cls_idx, score, name in human_yolo_detections:
                detections.append(
                    Detection(
                        bbox=bbox,
                        cls=CLASS_IDS["human"],
                        landing_status="-1",
                        motion_status="-1",
                        score=score,
                        name=name,
                    )
                )

            packet = make_frame_packet(frame_idx, detections)
            jsonl_file.write(json.dumps(packet, ensure_ascii=False) + "\n")
            jsonl_file.flush()

            if args.save_frame_json:
                (frame_json_dir / f"frame_{frame_idx:06d}.json").write_text(
                    json.dumps(packet, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            if annotated_writer is not None:
                annotated_writer.write(draw_annotations(frame, detections))

            processed_count += 1
            prev_frame = frame

            if processed_count % 25 == 0:
                elapsed = max(1e-6, time.time() - started_at)
                print(f"[INFO] Islenen frame: {processed_count} | Son frame id: {frame_idx:06d} | {processed_count / elapsed:.2f} fps")

            if args.max_frames is not None and processed_count >= args.max_frames:
                break

    capture.release()
    if annotated_writer is not None:
        annotated_writer.release()

    elapsed = time.time() - started_at
    summary = {
        "video": str(args.video),
        "results_dir": str(args.results_dir),
        "processed_frames": processed_count,
        "seen_frames": seen_count,
        "total_frames": total_frames,
        "source_fps": source_fps,
        "target_fps": args.target_fps,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "model_confs": {
            "landing": effective_model_conf(args, "landing"),
            "vehicle": effective_model_conf(args, "vehicle"),
            "human": effective_model_conf(args, "human"),
        },
        "iou": args.iou,
        "device": f"cuda:{yolo_device}" if yolo_device != "cpu" else "cpu",
        "classifier_positive_index": positive_index,
        "elapsed_seconds": elapsed,
        "predictions_jsonl": str(jsonl_path),
        "frame_json_dir": str(frame_json_dir) if args.save_frame_json else None,
        "annotated_video": str(args.results_dir / "annotated_prova.mp4") if args.save_annotated_video else None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Islenen frame: {processed_count}")
    print(f"[DONE] Ozet: {summary_path}")


def load_runtime(args: argparse.Namespace) -> tuple[str, torch.device, int, YOLO, YOLO, YOLO, torch.nn.Module, transforms.Compose, LandingReferenceMatcher | None]:
    require_paths([args.human_model, args.vehicle_model, args.landing_model, args.landing_classifier])
    yolo_device, torch_device = resolve_device(args.device)
    positive_index = 0 if args.flip_landing_classes else args.classifier_positive_index

    print(f"[INFO] Device: cuda:{yolo_device}" if yolo_device != "cpu" else "[INFO] Device: cpu")
    print("[INFO] Modeller yukleniyor...")
    landing_detector = YOLO(str(args.landing_model))
    vehicle_detector = YOLO(str(args.vehicle_model))
    human_detector = YOLO(str(args.human_model))
    landing_classifier = load_landing_classifier(args.landing_classifier, torch_device)
    classifier_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    landing_matcher = build_landing_reference_matcher(args, torch_device)
    return yolo_device, torch_device, positive_index, landing_detector, vehicle_detector, human_detector, landing_classifier, classifier_transform, landing_matcher


def run_video_mode(
    args: argparse.Namespace,
    runtime: tuple[str, torch.device, int, YOLO, YOLO, YOLO, torch.nn.Module, transforms.Compose, LandingReferenceMatcher | None],
    reference_predictions: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    require_paths([args.video])
    yolo_device, torch_device, positive_index, landing_detector, vehicle_detector, human_detector, landing_classifier, classifier_transform, landing_matcher = runtime

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Video acilamadi: {args.video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = args.target_fps if args.target_fps and args.target_fps > 0 else (source_fps or 25.0)

    if args.start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    annotated_writer = None
    if args.save_annotated_video:
        annotated_writer = open_video_writer(args.results_dir / "annotated_prova.mp4", output_fps, width, height)

    frame_json_dir = args.results_dir / "frame_json"
    if args.save_frame_json:
        frame_json_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = args.results_dir / "predictions.json"
    processed_count = 0
    seen_count = 0
    prediction_packets: list[dict[str, Any]] = []
    prev_frame: np.ndarray | None = None
    prev_tracks: list[VehicleTrack] = []
    started_at = time.time()

    print(f"[INFO] Video: {args.video}")
    print(f"[INFO] FPS: {source_fps:.3f} | Cozunurluk: {width}x{height} | Toplam frame: {total_frames}")
    print(f"[INFO] Prediction JSON cikti: {predictions_path}")

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_idx = args.start_frame + seen_count
        seen_count += 1
        if not should_process_frame(frame_idx, source_fps, args.target_fps):
            continue

        detections, prev_tracks = predict_frame(
            frame,
            frame_idx,
            args,
            yolo_device,
            torch_device,
            landing_detector,
            vehicle_detector,
            human_detector,
            landing_classifier,
            classifier_transform,
            positive_index,
            landing_matcher,
            prev_frame,
            prev_tracks,
        )
        frame_id = f"frame_{frame_idx:06d}"
        packet = make_frame_packet(
            frame_idx,
            detections,
            frame_id=frame_id,
            reference_objects=reference_predictions.get(frame_idx, []),
        )
        prediction_packets.append(packet)

        if args.save_frame_json:
            (frame_json_dir / f"frame_{frame_idx:06d}.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
        if annotated_writer is not None:
            annotated_writer.write(draw_annotations(frame, detections))

        processed_count += 1
        prev_frame = frame

        if processed_count % 25 == 0:
            elapsed = max(1e-6, time.time() - started_at)
            print(f"[INFO] Islenen frame: {processed_count} | Son frame id: {frame_idx:06d} | {processed_count / elapsed:.2f} fps")

        if args.max_frames is not None and processed_count >= args.max_frames:
            break

    write_predictions_json(predictions_path, prediction_packets)

    capture.release()
    if annotated_writer is not None:
        annotated_writer.release()

    return {
        "source": "video",
        "video": str(args.video),
        "processed_frames": processed_count,
        "seen_frames": seen_count,
        "total_frames": total_frames,
        "source_fps": source_fps,
        "target_fps": args.target_fps,
        "predictions_json": str(predictions_path),
        "frame_json_dir": str(frame_json_dir) if args.save_frame_json else None,
        "annotated_video": str(args.results_dir / "annotated_prova.mp4") if args.save_annotated_video else None,
    }


def run_dataset_mode(
    args: argparse.Namespace,
    runtime: tuple[str, torch.device, int, YOLO, YOLO, YOLO, torch.nn.Module, transforms.Compose, LandingReferenceMatcher | None],
    reference_predictions: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    annotation_paths = args.annotations if args.annotations else sorted(args.dataset_dir.glob("_annotations*.json"))
    require_paths([args.dataset_dir, *annotation_paths])

    yolo_device, torch_device, positive_index, landing_detector, vehicle_detector, human_detector, landing_classifier, classifier_transform, landing_matcher = runtime
    image_paths, ground_truths = load_coco_ground_truths(annotation_paths, args.dataset_dir)
    if args.start_frame > 0:
        image_paths = image_paths[args.start_frame :]
    if args.max_frames is not None:
        image_paths = image_paths[: args.max_frames]

    frame_json_dir = args.results_dir / "frame_json"
    annotated_dir = args.results_dir / "annotated_images"
    if args.save_frame_json:
        frame_json_dir.mkdir(parents=True, exist_ok=True)
    if args.save_annotated_images:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = args.results_dir / "predictions.json"
    predictions_for_eval: dict[str, list[Detection]] = {}
    prediction_packets: list[dict[str, Any]] = []
    processed_count = 0
    prev_frame: np.ndarray | None = None
    prev_tracks: list[VehicleTrack] = []
    started_at = time.time()

    print(f"[INFO] Dataset: {args.dataset_dir}")
    print(f"[INFO] Annotation dosyalari: {', '.join(p.name for p in annotation_paths)}")
    print(f"[INFO] Islenecek gorsel: {len(image_paths)}")
    print(f"[INFO] Prediction JSON cikti: {predictions_path}")

    for local_idx, image_path in enumerate(image_paths):
        frame_idx = args.start_frame + local_idx
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Gorsel okunamadi: {image_path}")
            continue

        detections, prev_tracks = predict_frame(
            frame,
            frame_idx,
            args,
            yolo_device,
            torch_device,
            landing_detector,
            vehicle_detector,
            human_detector,
            landing_classifier,
            classifier_transform,
            positive_index,
            landing_matcher,
            prev_frame,
            prev_tracks,
        )
        frame_id = canonical_frame_id(image_path.name, frame_idx)
        frame_number = extract_frame_number(frame_id)
        packet = make_frame_packet(
            frame_idx,
            detections,
            frame_id=frame_id,
            reference_objects=reference_predictions.get(frame_number, []) if frame_number is not None else [],
        )
        prediction_packets.append(packet)
        predictions_for_eval[image_path.name] = detections

        if args.save_frame_json:
            (frame_json_dir / f"{frame_id}.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.save_annotated_images:
            annotated = draw_ground_truth(draw_annotations(frame, detections), ground_truths.get(image_path.name, []))
            cv2.imwrite(str(annotated_dir / image_path.name), annotated)

        processed_count += 1
        prev_frame = frame

        if processed_count % 25 == 0:
            elapsed = max(1e-6, time.time() - started_at)
            print(f"[INFO] Islenen gorsel: {processed_count}/{len(image_paths)} | Son: {image_path.name} | {processed_count / elapsed:.2f} img/s")

    write_predictions_json(predictions_path, prediction_packets)

    evaluated_ground_truths = {image_path.name: ground_truths.get(image_path.name, []) for image_path in image_paths}
    eval_summary = evaluate_predictions(predictions_for_eval, evaluated_ground_truths, args.score_iou)
    eval_light = {
        "iou_threshold": eval_summary["iou_threshold"],
        "overall": eval_summary["overall"],
        "classes": {
            cls_id: {k: v for k, v in cls_data.items() if k != "matches"}
            for cls_id, cls_data in eval_summary["classes"].items()
        },
    }
    (args.results_dir / "evaluation_summary.json").write_text(json.dumps(eval_light, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.results_dir / "evaluation_matches.json").write_text(json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "source": "dataset",
        "dataset_dir": str(args.dataset_dir),
        "annotations": [str(p) for p in annotation_paths],
        "processed_frames": processed_count,
        "predictions_json": str(predictions_path),
        "frame_json_dir": str(frame_json_dir) if args.save_frame_json else None,
        "annotated_images_dir": str(annotated_dir) if args.save_annotated_images else None,
        "evaluation_summary": str(args.results_dir / "evaluation_summary.json"),
        "evaluation_matches": str(args.results_dir / "evaluation_matches.json"),
        "score": eval_light,
    }


def calculate_task1_points(run_summary: dict[str, Any], max_points: float) -> dict[str, Any] | None:
    score = run_summary.get("score")
    if not isinstance(score, dict):
        return None
    overall = score.get("overall")
    if not isinstance(overall, dict):
        return None
    map50 = overall.get("map50")
    if map50 is None:
        return None

    map50_float = float(map50)
    return {
        "metric": "mAP50",
        "metric_value": map50_float,
        "max_points": max_points,
        "points": map50_float * max_points,
        "percentage": map50_float * 100.0,
        "note": "COCO etiketlerinde landing_status ve motion_status olmadigi icin puan bbox+sinif mAP50 uzerinden hesaplandi.",
    }


def main2() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results: {args.results_dir}")
    started_at = time.time()
    reference_pipeline_payload: dict[str, Any] | None = None
    if not args.no_reference_objects and args.run_reference_pipeline:
        reference_pipeline_payload = run_reference_pipeline(args)

    runtime = load_runtime(args)
    reference_predictions: dict[int, list[dict[str, Any]]] = {}
    if not args.no_reference_objects:
        if reference_pipeline_payload is not None:
            reference_predictions = load_reference_predictions_from_payload(
                reference_pipeline_payload,
                args.reference_object_id,
                args.reference_min_final_score,
            )
        else:
            reference_predictions = load_reference_predictions(
                args.reference_predictions,
                args.reference_object_id,
                args.reference_min_final_score,
            )
        reference_count = sum(len(items) for items in reference_predictions.values())
        print(f"[INFO] Referans nesne tahmini: {reference_count} bbox / {len(reference_predictions)} frame")

    if args.source == "video":
        run_summary = run_video_mode(args, runtime, reference_predictions)
    else:
        run_summary = run_dataset_mode(args, runtime, reference_predictions)

    elapsed = time.time() - started_at
    task1_points = calculate_task1_points(run_summary, args.task1_max_points)
    reference_score: dict[str, Any] | None = None
    if not args.no_reference_objects and args.reference_gt.exists():
        evaluated_frames = [frame_number for frame_number, _name, _path in reference_candidate_images(args)]
        reference_gt = load_reference_ground_truths(args.reference_gt)
        reference_score_full = evaluate_reference_objects(
            reference_predictions,
            reference_gt,
            evaluated_frames,
            args.score_iou,
        )
        reference_score = {key: value for key, value in reference_score_full.items() if key != "matches"}
    reference_candidate_count = 0
    reference_detected_candidate_count = 0
    if reference_pipeline_payload is not None:
        reference_frames = reference_pipeline_payload.get("frames", [])
        reference_candidate_count = len(reference_frames)
        reference_detected_candidate_count = sum(1 for item in reference_frames if item.get("bbox"))
    run_summary.update(
        {
            "results_dir": str(args.results_dir),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "model_confs": {
                "landing": effective_model_conf(args, "landing"),
                "vehicle": effective_model_conf(args, "vehicle"),
                "human": effective_model_conf(args, "human"),
            },
            "iou": args.iou,
            "score_iou": args.score_iou,
            "task1_max_points": args.task1_max_points,
            "task1_points": task1_points,
            "reference_objects": {
                "enabled": not args.no_reference_objects,
                "pipeline_ran": reference_pipeline_payload is not None,
                "pipeline_output": "predictions.json:detected_undefined_objects" if reference_pipeline_payload is not None else None,
                "external_predictions": str(args.reference_predictions) if not args.no_reference_objects and reference_pipeline_payload is None else None,
                "object_id": args.reference_object_id if not args.no_reference_objects else None,
                "min_final_score": args.reference_min_final_score,
                "frame_count": len(reference_predictions),
                "bbox_count": sum(len(items) for items in reference_predictions.values()),
                "candidate_frame_count": reference_candidate_count,
                "candidate_bbox_count": reference_detected_candidate_count,
                "ground_truth": str(args.reference_gt) if args.reference_gt.exists() else None,
                "score": reference_score,
            },
            "landing_embedding_check": {
                "enabled": args.landing_embedding_check,
                "clip_model": args.clip_model if args.landing_embedding_check else None,
                "dinov2_model": args.dinov2_model if args.landing_embedding_check else None,
                "min_similarity": args.embedding_min_sim,
                "margin": args.embedding_margin,
                "clip_weight": args.embedding_clip_weight,
                "dino_weight": args.embedding_dino_weight,
                "reference_dir": str(args.reference_dir),
                "reference_samples_per_class": args.reference_samples_per_class,
                "manual_uap_references": [str(path) for path in args.uap_reference],
                "manual_uai_references": [str(path) for path in args.uai_reference],
            },
            "elapsed_seconds": elapsed,
        }
    )
    summary_path = args.results_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Islenen frame/gorsel: {run_summary.get('processed_frames')}")
    print(f"[DONE] Ozet: {summary_path}")


if __name__ == "__main__":
    main2()
